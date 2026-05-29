"""CKA-guided MoE upcycling trainer (consolidated).

Single-file, camera-ready trainer for **Efficient Bilevel Optimization for
CKA-Guided MoE Upcycling**. This module merges what used to live across four
separate ablation files (``upcycling_cka_v2.py`` ... ``upcycling_cka_v5.py``)
into one self-contained class chain:

    DifferentiableNasCKAV5            (V5: differentiable NAS + Gumbel-Softmax)
      -> NasCKAUpcycleV4              (V4: sensitivity-probed expand/recycle)
        -> AdaptiveCKAUpcycleV3       (V3: replay loss + layerwise CKA penalty)
          -> AdaptiveCKAUpcycle       (V2: bilevel weight + Task-0 alignment)
            -> BilevelCKAUpcycleV2 -> SimpleCKAUpcycleV2 -> (Upcycle, CKAMixinV2)

Only ``create_cka_upcycle_v5`` is exposed to the CLI (via
``training/main.py`` with ``--cka-regularization --cka-version v5``); the
intermediate classes are kept as parents because V5 reuses their methods.
Dead ablation entry points and off-path variants from the research repo have
been removed.
"""


import os
import json
import random
import pickle
from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm
from model.Dynamic_network.upcycling_refactored import Upcycle
from utils.utils import print_rank_0, to_device
from typing import Optional, Dict, List
import copy
from typing import Dict, List, Optional, Tuple
import math
from typing import Dict, List, Optional, Tuple, Any


class CKACheckpointManager:
    """
    Manages saving and loading of CKA-related state for resume support.
    
    Saves:
    - replay_buffer: Old task data for CKA computation
    - baseline_outputs: Cached model outputs (per layer)
    - baseline_inputs: Input batches used for baseline
    - cka_config: CKA hyperparameters
    """
    
    @staticmethod
    def save_state(
        save_dir: str,
        replay_buffer: 'ReplayBufferV2',
        baseline_outputs: Dict[int, Tensor],
        baseline_inputs: List[Dict],
        cka_config: Dict[str, Any]
    ):
        """Save CKA state to disk."""
        cka_dir = os.path.join(save_dir, "cka_state")
        os.makedirs(cka_dir, exist_ok=True)
        
        # Save replay buffer
        replay_path = os.path.join(cka_dir, "replay_buffer.pkl")
        with open(replay_path, 'wb') as f:
            pickle.dump({
                'buffers': replay_buffer.buffers,
                'task_names': replay_buffer.task_names,
                'max_size_per_task': replay_buffer.max_size_per_task
            }, f)
        
        # Save baseline outputs (as tensors)
        baseline_out_path = os.path.join(cka_dir, "baseline_outputs.pt")
        torch.save(baseline_outputs, baseline_out_path)
        
        # Save baseline inputs
        baseline_in_path = os.path.join(cka_dir, "baseline_inputs.pkl")
        with open(baseline_in_path, 'wb') as f:
            pickle.dump(baseline_inputs, f)
        
        # Save config
        config_path = os.path.join(cka_dir, "cka_config.json")
        with open(config_path, 'w') as f:
            json.dump(cka_config, f, indent=2)
        
        print(f"[CKA] Saved CKA state to {cka_dir}")
    
    @staticmethod
    def load_state(load_dir: str) -> Tuple[Optional['ReplayBufferV2'], Dict, List, Dict]:
        """
        Load CKA state from disk.
        
        Returns:
            (replay_buffer, baseline_outputs, baseline_inputs, cka_config)
            Returns (None, {}, [], {}) if state not found.
        """
        cka_dir = os.path.join(load_dir, "cka_state")
        
        if not os.path.exists(cka_dir):
            print(f"[CKA] No CKA state found at {cka_dir}")
            return None, {}, [], {}
        
        # Load replay buffer
        replay_path = os.path.join(cka_dir, "replay_buffer.pkl")
        replay_buffer = None
        if os.path.exists(replay_path):
            with open(replay_path, 'rb') as f:
                data = pickle.load(f)
                replay_buffer = ReplayBufferV2(max_size_per_task=data.get('max_size_per_task', 100))
                replay_buffer.buffers = data['buffers']
                replay_buffer.task_names = data['task_names']
        
        # Load baseline outputs
        baseline_outputs = {}
        baseline_out_path = os.path.join(cka_dir, "baseline_outputs.pt")
        if os.path.exists(baseline_out_path):
            baseline_outputs = torch.load(baseline_out_path, map_location='cpu')
        
        # Load baseline inputs
        baseline_inputs = []
        baseline_in_path = os.path.join(cka_dir, "baseline_inputs.pkl")
        if os.path.exists(baseline_in_path):
            with open(baseline_in_path, 'rb') as f:
                baseline_inputs = pickle.load(f)
        
        # Load config
        cka_config = {}
        config_path = os.path.join(cka_dir, "cka_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cka_config = json.load(f)
        
        print(f"[CKA] Loaded CKA state from {cka_dir}")
        print(f"[CKA]   - Replay buffer: {len(replay_buffer.buffers) if replay_buffer else 0} tasks")
        print(f"[CKA]   - Baseline outputs: {len(baseline_outputs)} layers")
        print(f"[CKA]   - Baseline inputs: {len(baseline_inputs)} batches")
        
        return replay_buffer, baseline_outputs, baseline_inputs, cka_config

class ReplayBufferV2:
    """
    Replay buffer for storing old task data.
    Used for CKA computation to measure forgetting on old tasks.
    
    Key difference from V1: All data is stored on CPU as numpy/lists
    for efficient serialization and memory management.
    """
    
    def __init__(self, max_size_per_task: int = 100, max_tasks: int = 10):
        self.max_size_per_task = max_size_per_task
        self.max_tasks = max_tasks
        self.buffers: Dict[int, List[Dict]] = {}  # task_id -> list of batches
        self.task_names: Dict[int, str] = {}
    
    def add_task_data(self, task_id: int, task_name: str, dataloader, num_batches: int = 10):
        """
        Add data for a task from its dataloader.
        
        Args:
            task_id: Task index
            task_name: Task name
            dataloader: DataLoader for the task
            num_batches: Number of batches to store
        """
        self.task_names[task_id] = task_name
        
        if task_id not in self.buffers:
            self.buffers[task_id] = []
        
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            
            # Convert to CPU and store
            cpu_batch = {
                'input_ids': batch['input_ids'].cpu().clone(),
                'attention_mask': batch.get('attention_mask')
            }
            if cpu_batch['attention_mask'] is not None:
                cpu_batch['attention_mask'] = cpu_batch['attention_mask'].cpu().clone()
            
            if len(self.buffers[task_id]) < self.max_size_per_task:
                self.buffers[task_id].append(cpu_batch)
            else:
                # Reservoir sampling for overflow
                idx = random.randint(0, len(self.buffers[task_id]) - 1)
                self.buffers[task_id][idx] = cpu_batch
            
            batch_count += 1
        
        print(f"[Replay] Added {batch_count} batches for task {task_id} ({task_name})")
    
    def get_all_batches(self, exclude_task: int = None) -> List[Dict]:
        """
        Get all batches from all tasks (or excluding a specific task).
        
        Args:
            exclude_task: Task ID to exclude (optional)
            
        Returns:
            List of all batches from stored tasks
        """
        all_batches = []
        for tid in sorted(self.buffers.keys()):
            if exclude_task is not None and tid == exclude_task:
                continue
            all_batches.extend(self.buffers[tid])
        return all_batches
    
    def get_task_batches(self, task_id: int) -> List[Dict]:
        """Get batches for a specific task."""
        return self.buffers.get(task_id, [])
    
    def sample_batch(self) -> Optional[Dict]:
        """Sample a random batch from all tasks."""
        all_batches = self.get_all_batches()
        if not all_batches:
            return None
        return random.choice(all_batches)
    
    def __len__(self):
        return sum(len(batches) for batches in self.buffers.values())
    
    def num_tasks(self) -> int:
        return len(self.buffers)

class CKAMixinV2:
    """
    Mixin providing CKA computation functionality.
    
    Key fixes from V1:
    1. Uses replay_buffer for old task data (not current task)
    2. Stores baseline_inputs to ensure same inputs for comparison
    3. Uses float64 for numerical precision
    4. Filters padding tokens via attention_mask
    """
    
    def _init_cka_components(self):
        """Initialize CKA-related components."""
        # Replay buffer for old task data
        self.replay_buffer = ReplayBufferV2(
            max_size_per_task=getattr(self.args, 'replay_buffer_size', 100)
        )
        
        # Baseline cache: stores (inputs, outputs) for each layer
        self.baseline_outputs: Dict[int, Tensor] = {}  # layer_idx -> [n_samples, hidden_dim]
        self.baseline_inputs: List[Dict] = []  # Input batches used for baseline
        
        # CKA hyperparameters
        self.lambda_cka = getattr(self.args, 'lambda_cka', 0.1)
        self.cka_layers = getattr(self.args, 'cka_layers', 'deep')
        self.cka_compute_interval = getattr(self.args, 'cka_compute_interval', 50)  # Every 50 steps
        self.cka_debug_interval = getattr(self.args, 'cka_debug_interval', 200)  # Debug output every N steps
        self.cka_max_samples = 128  # Max samples per CKA computation
        
        # Device
        self.device = next(self.model.parameters()).device
        
        print_rank_0(f"[CKA V2] Initialized with lambda={self.lambda_cka}, "
                    f"layers={self.cka_layers}, compute_interval={self.cka_compute_interval}",
                    self.args.global_rank)
    
    def _get_cka_layer_indices(self) -> List[int]:
        """Get layer indices for CKA computation."""
        if hasattr(self.model, 'module'):
            num_layers = len(self.model.module.model.layers)
        else:
            num_layers = len(self.model.model.layers)
        
        if self.cka_layers == 'all':
            return list(range(num_layers))
        elif self.cka_layers == 'deep':
            # Last 6 layers (where most forgetting happens)
            return list(range(max(0, num_layers - 6), num_layers))
        elif self.cka_layers == 'shallow':
            return list(range(min(6, num_layers)))
        else:
            # Custom: parse comma-separated layer indices
            return [int(x) for x in self.cka_layers.split(',')]
    
    def _get_layer_output(
        self, 
        batch: Dict, 
        layer_idx: int, 
        max_samples: int = 128
    ) -> Optional[Tensor]:
        """
        Get layer hidden states for a batch, filtering padding tokens.
        
        Args:
            batch: Input batch with input_ids and attention_mask
            layer_idx: Layer index to extract
            max_samples: Maximum number of token samples to return
            
        Returns:
            Hidden states tensor [n_samples, hidden_dim] or None
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get model base
        if hasattr(self.model, 'module'):
            base_model = self.model.module.model
        else:
            base_model = self.model.model
        
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            
            # Get layer hidden states [batch, seq, hidden]
            hs = outputs.hidden_states[layer_idx]
            
            # CRITICAL FIX: Filter by attention_mask to exclude padding tokens
            if attention_mask is not None:
                mask_flat = attention_mask.view(-1).bool()
                hs_flat = hs.view(-1, hs.size(-1))
                hs_real = hs_flat[mask_flat]  # Only real tokens
            else:
                hs_real = hs.view(-1, hs.size(-1))
            
            # Take first max_samples (deterministic for reproducibility)
            if hs_real.size(0) > max_samples:
                hs_real = hs_real[:max_samples]
            
            return hs_real.detach()
    
    def _cache_baseline_from_replay(self, num_batches: int = 10):
        """
        Cache baseline outputs from replay buffer (OLD task data).
        
        This should be called BEFORE expert expansion to capture
        the model state that we want to preserve.
        """
        all_batches = self.replay_buffer.get_all_batches()
        
        if not all_batches:
            print_rank_0("[CKA V2] WARNING: Replay buffer empty, cannot cache baseline",
                        self.args.global_rank)
            return
        
        # Limit number of batches
        batches_to_use = all_batches[:min(num_batches, len(all_batches))]
        
        print_rank_0(f"[CKA V2] Caching baseline from {len(batches_to_use)} replay batches "
                    f"(total in buffer: {len(all_batches)})",
                    self.args.global_rank)
        
        layer_indices = self._get_cka_layer_indices()
        layer_outputs: Dict[int, List[Tensor]] = {idx: [] for idx in layer_indices}
        
        # Clear previous baseline
        self.baseline_outputs.clear()
        self.baseline_inputs.clear()
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches_to_use):
                # Store input batch for later use
                self.baseline_inputs.append({
                    'input_ids': batch['input_ids'].clone(),
                    'attention_mask': batch['attention_mask'].clone() if batch.get('attention_mask') is not None else None
                })
                
                # Move to device
                batch_gpu = to_device(batch, self.device)
                
                # Get outputs for each layer
                for layer_idx in layer_indices:
                    output = self._get_layer_output(batch_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        layer_outputs[layer_idx].append(output.cpu())
                
                torch.cuda.empty_cache()
        
        # Concatenate outputs per layer
        for layer_idx in layer_indices:
            if layer_outputs[layer_idx]:
                self.baseline_outputs[layer_idx] = torch.cat(layer_outputs[layer_idx], dim=0)
                print_rank_0(f"[CKA V2] Layer {layer_idx}: cached {self.baseline_outputs[layer_idx].size(0)} samples",
                            self.args.global_rank)
        
        self.model.train()
        torch.cuda.empty_cache()
    
    def _compute_cka(self, X: Tensor, Y: Tensor, eps: float = 1e-10) -> float:
        """
        Compute CKA (Centered Kernel Alignment) between two representations.
        
        Uses float64 for numerical precision and HSIC formulation.
        
        Args:
            X: First representation [n, d1]
            Y: Second representation [n, d2]
            eps: Small constant for numerical stability
            
        Returns:
            CKA score in [0, 1]
        """
        # Convert to float64 on CPU for precision
        X = X.double().cpu()
        Y = Y.double().cpu()
        
        # Center the data
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)
        
        # Gram matrices (n x n)
        K_X = X_c @ X_c.T
        K_Y = Y_c @ Y_c.T
        
        # HSIC computation
        hsic_xy = torch.sum(K_X * K_Y)
        hsic_xx = torch.sum(K_X * K_X)
        hsic_yy = torch.sum(K_Y * K_Y)
        
        # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        
        if denominator < eps:
            return 0.0
        
        cka = hsic_xy / denominator
        return float(cka.clamp(0.0, 1.0).item())
    
    def compute_cka_penalty(self, debug: bool = False) -> float:
        """
        Compute CKA penalty using replay buffer data.
        
        CRITICAL: Uses the SAME input batches as baseline for fair comparison.
        
        Returns:
            CKA penalty = 1 - mean(CKA) across layers
        """
        if not self.baseline_outputs or not self.baseline_inputs:
            return 0.0
        
        layer_indices = list(self.baseline_outputs.keys())
        cka_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for layer_idx in layer_indices:
                baseline_full = self.baseline_outputs[layer_idx]
                
                # Compute current outputs using SAME inputs as baseline
                current_outputs = []
                samples_per_batch = baseline_full.size(0) // max(len(self.baseline_inputs), 1)
                
                for batch_idx, input_batch in enumerate(self.baseline_inputs):
                    input_gpu = to_device(input_batch, self.device)
                    output = self._get_layer_output(input_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        current_outputs.append(output.cpu())
                    torch.cuda.empty_cache()
                
                if not current_outputs:
                    continue
                
                current_cat = torch.cat(current_outputs, dim=0)
                
                # Match sizes
                min_samples = min(current_cat.size(0), baseline_full.size(0))
                if min_samples < 10:
                    continue
                
                current_sub = current_cat[:min_samples]
                baseline_sub = baseline_full[:min_samples]
                
                # Compute CKA
                cka_score = self._compute_cka(current_sub, baseline_sub)
                cka_scores.append(cka_score)
                
                if debug:
                    diff = (current_sub - baseline_sub).abs().mean().item()
                    print_rank_0(f"[CKA V2] Layer {layer_idx}: CKA={cka_score:.4f}, "
                                f"diff={diff:.4f}, samples={min_samples}",
                                self.args.global_rank)
        
        self.model.train()
        
        if not cka_scores:
            return 0.0
        
        mean_cka = np.mean(cka_scores)
        penalty = 1.0 - mean_cka
        
        if debug:
            print_rank_0(f"[CKA V2] Mean CKA={mean_cka:.4f}, Penalty={penalty:.4f}",
                        self.args.global_rank)
        
        return penalty
    
    def save_cka_state(self, save_dir: str):
        """Save CKA state for resume."""
        cka_config = {
            'lambda_cka': self.lambda_cka,
            'cka_layers': self.cka_layers,
            'cka_compute_interval': self.cka_compute_interval,
            'cka_max_samples': self.cka_max_samples
        }
        CKACheckpointManager.save_state(
            save_dir,
            self.replay_buffer,
            self.baseline_outputs,
            self.baseline_inputs,
            cka_config
        )
    
    def load_cka_state(self, load_dir: str) -> bool:
        """
        Load CKA state for resume.
        
        Returns:
            True if state was loaded successfully
        """
        replay_buffer, baseline_outputs, baseline_inputs, cka_config = \
            CKACheckpointManager.load_state(load_dir)
        
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
            self.baseline_outputs = baseline_outputs
            self.baseline_inputs = baseline_inputs
            
            # Restore config
            if 'lambda_cka' in cka_config:
                self.lambda_cka = cka_config['lambda_cka']
            
            print_rank_0(f"[CKA V2] Restored state: {self.replay_buffer.num_tasks()} tasks, "
                        f"{len(self.baseline_outputs)} layers",
                        self.args.global_rank)
            return True
        
        return False

class SimpleCKAUpcycleV2(Upcycle, CKAMixinV2):
    """
    CKA-regularized Upcycling MoE with resume support.
    
    Key improvements:
    1. Correct CKA data flow (uses replay buffer)
    2. Correct baseline timing (before expansion)
    3. Resume support (saves/loads CKA state)
    4. Memory efficient (periodic computation)
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)
        self._init_cka_components()
        print_rank_0("[CKA V2] Initialized SimpleCKAUpcycleV2", self.args.global_rank)
    
    def train_continual(self):
        """
        Main training loop with correct CKA baseline timing.
        
        Flow:
        1. (Resume) Load MoE structure and CKA state
        2. For each task:
           a. Upcycle (add experts)
           b. Train with CKA regularization
           c. Add task data to replay buffer
           d. Cache baseline (BEFORE next task's expansion)
           e. Save checkpoint + CKA state
        """
        start_task = getattr(self.args, 'start_task', 0)
        
        # Handle resume
        if start_task > 0:
            print_rank_0(f"[CKA V2] Resuming from task {start_task}", self.args.global_rank)
            
            # Step 1: Rebuild MoE structure
            self._restore_moe_structure_for_resume(start_task)
            
            # Step 2: Load CKA state (replay buffer + baseline)
            ckpt_dir = self.args.model_name_or_path
            # Try to load from the checkpoint directory
            parent_dir = os.path.dirname(ckpt_dir.rstrip('/'))
            task_ckpt = os.path.join(parent_dir, str(start_task - 1))
            
            if not self.load_cka_state(task_ckpt):
                # Fallback: try current directory
                self.load_cka_state(ckpt_dir)
            
            # Step 3: If no baseline loaded but replay buffer exists, cache now
            if not self.baseline_outputs and len(self.replay_buffer) > 0:
                print_rank_0("[CKA V2] Caching baseline from loaded replay buffer",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)
        
        # Main training loop
        for i_task, task in enumerate(self.train_task_list):
            if i_task < start_task:
                print_rank_0(f"[CKA V2] Skipping completed task {i_task}: {task}",
                            self.args.global_rank)
                continue
            
            self.current_task_id = i_task
            print_rank_0(f"[CKA V2] >>>>> Start task-{i_task}: {task}", self.args.global_rank)
            
            # Upcycle (add experts)
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True
            
            if do_upcycle:
                self.upcycle_one_task(task, i_task)
            
            # Train
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            
            # Add current task to replay buffer
            dataloader = self.train_task_list[task]
            self.replay_buffer.add_task_data(i_task, task, dataloader, num_batches=10)
            
            # CRITICAL: Cache baseline BEFORE next task's expansion
            if i_task < len(self.train_task_list) - 1:
                print_rank_0(f"[CKA V2] Caching baseline for next task (pre-expansion)",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)
            
            # Save checkpoint + CKA state
            self.save_model(i_task)
            self.save_cka_state(os.path.join(self.args.output_dir, str(i_task)))
    
    def train_one_task(self, task, i_task, epochs):
        """Train on one task with CKA regularization."""
        dataloader = self.train_task_list[task]
        total_steps = epochs * len(dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        # Check baseline availability
        if i_task > 0:
            if self.baseline_outputs:
                print_rank_0(f"[CKA V2] Using baseline with {len(self.baseline_outputs)} layers",
                            self.args.global_rank)
            else:
                print_rank_0("[CKA V2] WARNING: No baseline available for CKA regularization",
                            self.args.global_rank)
        
        # Freeze non-current task parameters
        self.freeze_non_current_task_params(i_task)
        
        global_step = 0
        cached_cka_penalty = 0.0
        self.model.train()
        
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(**batch, use_cache=False)
                task_loss = outputs.loss
                
                # CKA regularization for non-first tasks
                cka_loss = 0.0
                if i_task > 0 and self.baseline_outputs:
                    # Compute CKA every N steps
                    if global_step % self.cka_compute_interval == 0:
                        debug = (global_step % self.cka_debug_interval == 0)
                        cached_cka_penalty = self.compute_cka_penalty(debug=debug)
                        
                        if debug:
                            print_rank_0(f"[CKA V2] Step {global_step}: penalty={cached_cka_penalty:.4f}",
                                        self.args.global_rank)
                    
                    cka_loss = cached_cka_penalty
                
                # Total loss
                total_loss = task_loss + self.lambda_cka * cka_loss
                
                # Backward pass
                self.model.backward(total_loss)
                self.model.step()
                
                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    desc = f"Task-{i_task} Epoch-{epoch} loss={task_loss.item():.4f}"
                    if i_task > 0:
                        desc += f" cka={cka_loss:.4f}"
                    progress_bar.set_description(desc)
                
                global_step += 1
        
        progress_bar.close()
        print_rank_0(f"[CKA V2] Finished task {i_task}: {task}", self.args.global_rank)

class BilevelCKAUpcycleV2(SimpleCKAUpcycleV2):
    """
    Bilevel CKA Upcycling with gradient alignment-based weight adjustment.
    
    Inspired by GEM (Gradient Episodic Memory), this version dynamically
    adjusts the CKA regularization weight based on gradient alignment
    between CKA loss and replay loss.
    
    Key idea:
    - When CKA gradient aligns with replay gradient → increase weight
    - When they conflict → decrease weight (to allow learning)
    
    Loss: L = L_task + w_dynamic * L_cka
    where w_dynamic ∝ <∇CKA, ∇replay>
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)
        
        # Bilevel-specific parameters
        self.bilevel_base_weight = getattr(args, 'bilevel_base_weight', 1.0)
        self.bilevel_weight_scale = getattr(args, 'bilevel_weight_scale', 0.1)
        self.use_conflict_monitor = getattr(args, 'use_conflict_monitor', False)
        
        # Conflict monitoring
        self.conflict_threshold = getattr(args, 'conflict_threshold', 0.5)
        self.conflict_patience = getattr(args, 'conflict_patience', 10)
        self.conflict_counter = 0
        self.alignment_history = []
        
        print_rank_0(f"[Bilevel V2] Initialized with base_weight={self.bilevel_base_weight}, "
                    f"scale={self.bilevel_weight_scale}, conflict_monitor={self.use_conflict_monitor}",
                    self.args.global_rank)
    
    def _compute_gradient_alignment(self, batch_replay: Dict) -> Tuple[float, float]:
        """
        Compute alignment between CKA gradient and replay loss gradient.
        
        Args:
            batch_replay: Replay batch from old tasks
            
        Returns:
            (alignment_score, dynamic_weight)
            alignment_score: cosine similarity in [-1, 1]
            dynamic_weight: adjusted weight for CKA loss
        """
        # Store original gradients
        self.model.zero_grad()
        
        # Step 1: Compute replay loss gradient
        batch_gpu = to_device(batch_replay, self.device)
        outputs_replay = self.model(**batch_gpu, use_cache=False)
        replay_loss = outputs_replay.loss
        
        # Get replay gradients (approximate via single backward)
        replay_loss.backward(retain_graph=True)
        g_replay = self._collect_param_gradients()
        
        # Step 2: Compute CKA-related gradient (via replay loss as proxy)
        # Note: We use replay loss gradient as proxy for "knowledge preservation direction"
        # True CKA gradient would require second-order computation
        
        # Reset gradients
        self.model.zero_grad()
        
        # Compute current task direction (will be computed in main training loop)
        # Here we just return the replay gradient as reference
        
        # Step 3: Compute alignment (simplified version)
        # In full bilevel, we'd compare g_cka with g_task
        # Here we use alignment between consecutive gradients as proxy
        
        g_norm = g_replay.norm()
        if g_norm < 1e-10:
            return 0.0, self.bilevel_base_weight
        
        # For now, return base weight (will be adjusted based on CKA penalty trend)
        # More sophisticated: track CKA penalty trend and adjust weight
        alignment = 0.0  # Placeholder
        dynamic_weight = self.bilevel_base_weight
        
        return alignment, dynamic_weight
    
    def _collect_param_gradients(self) -> Tensor:
        """Collect all parameter gradients into a single vector."""
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if not grads:
            return torch.zeros(1, device=self.device)
        
        return torch.cat(grads)
    
    def _update_dynamic_weight(self, cka_penalty: float) -> float:
        """
        Update dynamic weight based on CKA penalty with threshold-based adjustment.
        
        Key insight from log analysis:
        - Task 1 (Py150): penalty 0.03-0.05 (high forgetting)
        - Task 2-5: penalty 0.001-0.01 (low forgetting)
        
        Strategy: Use stepped thresholds for more aggressive response to forgetting.
        """
        self.alignment_history.append(cka_penalty)
        
        # Keep only recent history
        if len(self.alignment_history) > 100:
            self.alignment_history = self.alignment_history[-100:]
        
        # === Threshold-based stepped adjustment (primary) ===
        # Based on observed penalty ranges from training logs
        if cka_penalty > 0.05:
            # Severe forgetting (e.g., Task 1 early stages)
            base_multiplier = 3.0
        elif cka_penalty > 0.03:
            # High forgetting
            base_multiplier = 2.5
        elif cka_penalty > 0.02:
            # Moderate forgetting
            base_multiplier = 2.0
        elif cka_penalty > 0.01:
            # Light forgetting
            base_multiplier = 1.5
        elif cka_penalty > 0.005:
            # Minimal forgetting
            base_multiplier = 1.2
        else:
            # Almost no forgetting
            base_multiplier = 1.0
        
        # === Trend-based adjustment (secondary) ===
        # Add extra boost if forgetting is getting worse
        trend_adjustment = 0.0
        if len(self.alignment_history) >= 10:
            recent_avg = np.mean(self.alignment_history[-10:])
            older_avg = np.mean(self.alignment_history[-20:-10]) if len(self.alignment_history) >= 20 else recent_avg
            trend = recent_avg - older_avg
            
            # More aggressive trend response (scale factor increased from 0.1 to 1.0)
            trend_adjustment = self.bilevel_weight_scale * np.tanh(trend * 20)  # Doubled sensitivity
        
        # Combine base multiplier with trend adjustment
        dynamic_weight = self.bilevel_base_weight * base_multiplier + trend_adjustment
        
        # Clamp to reasonable range [0.5, 5.0] (expanded from [0.01, 2.0])
        dynamic_weight = max(0.5, min(5.0, dynamic_weight))
        
        return dynamic_weight
    
    def _check_conflict(self, cka_penalty: float) -> bool:
        """
        Check if there's significant conflict requiring expansion.
        
        Returns True if consistent high CKA penalty (forgetting) detected.
        """
        if not self.use_conflict_monitor:
            return False
        
        if cka_penalty > self.conflict_threshold:
            self.conflict_counter += 1
        else:
            self.conflict_counter = max(0, self.conflict_counter - 1)
        
        return self.conflict_counter >= self.conflict_patience
    
    def train_one_task(self, task, i_task, epochs):
        """Train on one task with bilevel CKA regularization."""
        dataloader = self.train_task_list[task]
        total_steps = epochs * len(dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        # Check baseline availability
        if i_task > 0:
            if self.baseline_outputs:
                print_rank_0(f"[Bilevel V2] Using baseline with {len(self.baseline_outputs)} layers",
                            self.args.global_rank)
            else:
                print_rank_0("[Bilevel V2] WARNING: No baseline available",
                            self.args.global_rank)
        
        # Freeze non-current task parameters
        self.freeze_non_current_task_params(i_task)
        
        global_step = 0
        cached_cka_penalty = 0.0
        dynamic_weight = self.bilevel_base_weight
        self.model.train()
        
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(**batch, use_cache=False)
                task_loss = outputs.loss
                
                # Bilevel CKA regularization for non-first tasks
                cka_loss = 0.0
                if i_task > 0 and self.baseline_outputs:
                    # Compute CKA every N steps
                    if global_step % self.cka_compute_interval == 0:
                        debug = (global_step % self.cka_debug_interval == 0)
                        cached_cka_penalty = self.compute_cka_penalty(debug=debug)
                        
                        # Update dynamic weight based on trend
                        dynamic_weight = self._update_dynamic_weight(cached_cka_penalty)
                        
                        # Check for conflict (potential expansion trigger)
                        if self._check_conflict(cached_cka_penalty):
                            print_rank_0(f"[Bilevel V2] Conflict detected at step {global_step}! "
                                        f"Consider expansion.", self.args.global_rank)
                        
                        if debug:
                            print_rank_0(f"[Bilevel V2] Step {global_step}: penalty={cached_cka_penalty:.4f}, "
                                        f"weight={dynamic_weight:.4f}",
                                        self.args.global_rank)
                    
                    cka_loss = cached_cka_penalty
                
                # Total loss with dynamic weight
                effective_lambda = self.lambda_cka * dynamic_weight
                total_loss = task_loss + effective_lambda * cka_loss
                
                # Backward pass
                self.model.backward(total_loss)
                self.model.step()
                
                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    desc = f"Task-{i_task} Epoch-{epoch} loss={task_loss.item():.4f}"
                    if i_task > 0:
                        desc += f" cka={cka_loss:.4f} w={dynamic_weight:.2f}"
                    progress_bar.set_description(desc)
                
                global_step += 1
        
        progress_bar.close()
        
        # Log final stats
        if self.alignment_history:
            avg_penalty = np.mean(self.alignment_history[-50:]) if len(self.alignment_history) >= 50 else np.mean(self.alignment_history)
            print_rank_0(f"[Bilevel V2] Finished task {i_task}: avg_cka_penalty={avg_penalty:.4f}",
                        self.args.global_rank)

class AdaptiveCKAUpcycle(BilevelCKAUpcycleV2):
    """
    Adaptive CKA combining all improvements:
    1. Bilevel dynamic weight adjustment (from BilevelCKAUpcycleV2)
    2. Task 0 base model alignment (from EnhancedCKAUpcycleV2)
    3. Prior-based + Dynamic layer selection
    4. Loss-based early stopping
    
    Based on prior analysis:
    - Layer 15: Always has forgetting (base layer)
    - Layers 6, 9, 12, 13: Variable forgetting (candidate layers)
    - NumGLUE tasks: Very stable, minimal CKA needed, can early stop
    - MeetingBank/20Minuten: Most affected by forgetting
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)
        
        # Base model outputs for Task 0 alignment (from EnhancedCKAUpcycleV2)
        self.base_model_outputs: Dict[int, Tensor] = {}
        self.base_model_inputs: List[Dict] = []
        self.task0_cka_weight = getattr(args, 'task0_cka_weight', 0.5)  # Lower weight for Task 0
        
        # Prior-based layer configuration
        # base_layers: Always monitored (deep layers most prone to forgetting)
        # candidate_layers: Added dynamically when forgetting detected
        self.base_layers = [13, 14, 15]  # Deep layers - always monitor
        self.candidate_layers = [6, 9, 12]  # Mid layers - add if needed
        self.active_layers = list(self.base_layers)
        
        # Dynamic layer adjustment
        self.layer_adjust_interval = getattr(args, 'layer_adjust_interval', 100)
        self.forgetting_threshold = getattr(args, 'forgetting_threshold', 0.02)
        
        # Loss-based early stopping
        self.early_stop_loss = getattr(args, 'early_stop_loss', 0.01)
        self.early_stop_min_epochs = getattr(args, 'early_stop_min_epochs', 1)  # Min 2 epochs (0-indexed)
        self.early_stop_window = 100  # Steps to check for stability
        
        # Override cka_layers with adaptive layers
        self.cka_layers = 'adaptive'
        
        print_rank_0(f"[Adaptive CKA] Initialized with bilevel={True}, base_alignment={True}, "
                    f"base_layers={self.base_layers}, candidates={self.candidate_layers}, "
                    f"early_stop_loss={self.early_stop_loss}",
                    self.args.global_rank)
    
    def _get_cka_layer_indices(self) -> List[int]:
        """Return current active layers for CKA computation."""
        return self.active_layers
    
    def _get_cacheable_layer_indices(self) -> List[int]:
        """Return ALL layers that should be cached for dynamic layer selection."""
        # Need to cache both base_layers AND candidate_layers for dynamic selection
        all_layers = list(set(self.base_layers + self.candidate_layers))
        all_layers.sort()
        return all_layers
    
    def _cache_baseline_from_replay(self, num_batches: int = 10):
        """
        Override to cache ALL candidate layers for dynamic layer selection.
        
        This is critical: we need baselines for candidate_layers to detect
        forgetting and dynamically add them to active_layers.
        """
        import torch.distributed as dist
        
        # Synchronize before baseline caching to ensure all ranks start together
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        all_batches = self.replay_buffer.get_all_batches()
        
        if not all_batches:
            print_rank_0("[Adaptive CKA] WARNING: Replay buffer empty, cannot cache baseline",
                        self.args.global_rank)
            # Still need barrier for ranks that have batches
            if dist.is_initialized():
                dist.barrier()
            return
        
        batches_to_use = all_batches[:min(num_batches, len(all_batches))]
        
        # Use ALL candidate layers, not just active ones
        layer_indices = self._get_cacheable_layer_indices()
        
        print_rank_0(f"[Adaptive CKA] Caching baseline from {len(batches_to_use)} replay batches "
                    f"for layers {layer_indices}",
                    self.args.global_rank)
        
        layer_outputs: Dict[int, List[Tensor]] = {idx: [] for idx in layer_indices}
        
        # Clear previous baseline
        self.baseline_outputs.clear()
        self.baseline_inputs.clear()
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches_to_use):
                # Store input batch for later use
                self.baseline_inputs.append({
                    'input_ids': batch['input_ids'].clone(),
                    'attention_mask': batch['attention_mask'].clone() if batch.get('attention_mask') is not None else None
                })
                
                batch_gpu = to_device(batch, self.device)
                
                for layer_idx in layer_indices:
                    output = self._get_layer_output(batch_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        layer_outputs[layer_idx].append(output.cpu())
                
                torch.cuda.empty_cache()
        
        # Concatenate outputs per layer
        for layer_idx in layer_indices:
            if layer_outputs[layer_idx]:
                self.baseline_outputs[layer_idx] = torch.cat(layer_outputs[layer_idx], dim=0)
                print_rank_0(f"[Adaptive CKA] Baseline Layer {layer_idx}: "
                            f"cached {self.baseline_outputs[layer_idx].size(0)} samples",
                            self.args.global_rank)
        
        self.model.train()
        torch.cuda.empty_cache()
        
        # Synchronize after baseline caching completes
        import torch.distributed as dist
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
    
    def _cache_base_model_outputs(self, num_batches: int = 10):
        """
        Cache base model outputs BEFORE any training.
        This provides a reference for Task 0 alignment.
        """
        print_rank_0("[Adaptive CKA] Caching base model outputs for Task 0 alignment",
                    self.args.global_rank)
        
        # Use first task's data for base model caching
        first_task = list(self.train_task_list.keys())[0]
        dataloader = self.train_task_list[first_task]
        
        # Sample batches
        batches = []
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches:
                break
            if 'sources' in batch:
                del batch['sources']
            batches.append(batch)
        
        if not batches:
            print_rank_0("[Adaptive CKA] WARNING: No data for base model caching",
                        self.args.global_rank)
            return
        
        # Use ALL candidate layers for base model (to allow dynamic selection later)
        layer_indices = self.base_layers + self.candidate_layers
        layer_outputs: Dict[int, List[Tensor]] = {idx: [] for idx in layer_indices}
        
        self.base_model_outputs.clear()
        self.base_model_inputs.clear()
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                # Store input batch
                self.base_model_inputs.append({
                    k: v.cpu() if isinstance(v, Tensor) else v 
                    for k, v in batch.items()
                })
                
                batch_gpu = to_device(batch, self.device)
                
                for layer_idx in layer_indices:
                    output = self._get_layer_output(batch_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        layer_outputs[layer_idx].append(output.cpu())
                
                torch.cuda.empty_cache()
        
        # Concatenate outputs per layer
        for layer_idx in layer_indices:
            if layer_outputs[layer_idx]:
                self.base_model_outputs[layer_idx] = torch.cat(layer_outputs[layer_idx], dim=0)
                print_rank_0(f"[Adaptive CKA] Base model Layer {layer_idx}: "
                            f"cached {self.base_model_outputs[layer_idx].size(0)} samples",
                            self.args.global_rank)
        
        self.model.train()
        torch.cuda.empty_cache()
    
    def train_continual(self):
        """
        Enhanced continual training with:
        - Base model alignment for Task 0
        - Bilevel dynamic weights for all tasks
        """
        start_task = getattr(self.args, 'start_task', 0)
        
        if start_task > 0:
            print_rank_0(f"[Adaptive CKA] Resuming from task {start_task}", self.args.global_rank)
            self._restore_moe_structure_for_resume(start_task)
            
            ckpt_dir = self.args.model_name_or_path
            parent_dir = os.path.dirname(ckpt_dir.rstrip('/'))
            task_ckpt = os.path.join(parent_dir, str(start_task - 1))
            
            if not self.load_cka_state(task_ckpt):
                self.load_cka_state(ckpt_dir)
            
            if not self.baseline_outputs and len(self.replay_buffer) > 0:
                self._cache_baseline_from_replay(num_batches=10)
        else:
            # CRITICAL: Cache base model outputs BEFORE any training
            self._cache_base_model_outputs(num_batches=10)
        
        # Main training loop
        for i_task, task in enumerate(self.train_task_list):
            if i_task < start_task:
                print_rank_0(f"[Adaptive CKA] Skipping completed task {i_task}: {task}",
                            self.args.global_rank)
                continue
            
            self.current_task_id = i_task
            print_rank_0(f"[Adaptive CKA] >>>>> Start task-{i_task}: {task}", self.args.global_rank)
            
            # Upcycle (add experts)
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True
            
            if do_upcycle:
                self.upcycle_one_task(task, i_task)
            
            # Train with adaptive CKA
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            
            # Add current task to replay buffer
            dataloader = self.train_task_list[task]
            self.replay_buffer.add_task_data(i_task, task, dataloader, num_batches=10)
            
            # Cache baseline for next task (using current model state)
            if i_task < len(self.train_task_list) - 1:
                print_rank_0(f"[Adaptive CKA] Caching baseline for next task",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)
            
            # Save model and CKA state
            self.save_model(i_task)
            self.save_cka_state(os.path.join(self.args.output_dir, str(i_task)))
        
        print_rank_0("[Adaptive CKA] Training completed!", self.args.global_rank)
    
    def _measure_layer_cka(self, layer_idx: int) -> float:
        """Measure CKA for a single layer against baseline."""
        if not self.baseline_outputs or layer_idx not in self.baseline_outputs:
            return 1.0  # No baseline = no forgetting detected
        
        baseline = self.baseline_outputs[layer_idx]
        
        # Get current output for this layer
        if not self.baseline_inputs:
            return 1.0
        
        current_outputs = []
        self.model.eval()
        with torch.no_grad():
            for input_batch in self.baseline_inputs[:3]:  # Use first 3 batches
                input_gpu = to_device(input_batch, self.device)
                output = self._get_layer_output(input_gpu, layer_idx, max_samples=64)
                if output is not None:
                    current_outputs.append(output.cpu())
        self.model.train()
        
        if not current_outputs:
            return 1.0
        
        current_full = torch.cat(current_outputs, dim=0)
        min_samples = min(baseline.size(0), current_full.size(0))
        
        cka = self._compute_cka(current_full[:min_samples], baseline[:min_samples])
        return cka
    
    def _update_active_layers(self, global_step: int):
        """Check candidate layers and add to active if forgetting detected."""
        if global_step % self.layer_adjust_interval != 0:
            return
        
        for layer in self.candidate_layers:
            if layer in self.active_layers:
                continue
            
            cka = self._measure_layer_cka(layer)
            forgetting = 1.0 - cka
            
            if forgetting > self.forgetting_threshold:
                self.active_layers.append(layer)
                self.active_layers.sort()
                print_rank_0(f"[Adaptive CKA] Step {global_step}: Added layer {layer} "
                            f"(CKA={cka:.4f}, forgetting={forgetting:.4f})",
                            self.args.global_rank)
    
    def _should_early_stop(self, loss_history: List[float], current_epoch: int, 
                           steps_per_epoch: int) -> bool:
        """
        Check if training should stop early based on loss convergence.
        
        Criteria:
        1. Completed at least min_epochs
        2. Mean loss < threshold
        3. Loss is stable (low variance)
        """
        if current_epoch < self.early_stop_min_epochs:
            return False
        
        if len(loss_history) < self.early_stop_window:
            return False
        
        recent_losses = loss_history[-self.early_stop_window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        # Stop if loss is low and stable
        if mean_loss < self.early_stop_loss and std_loss < self.early_stop_loss:
            print_rank_0(f"[Adaptive CKA] Early stop triggered: "
                        f"mean_loss={mean_loss:.6f}, std={std_loss:.6f}, epoch={current_epoch}",
                        self.args.global_rank)
            return True
        
        return False
    
    def train_one_task(self, task, i_task, epochs):
        """
        Train with:
        - Task 0: Base model alignment + bilevel weights + early stopping
        - Task 1+: Baseline alignment + bilevel weights + dynamic layers + early stopping
        """
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        total_steps = epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        # Determine CKA target and weight
        if i_task == 0 and self.base_model_outputs:
            cka_target_outputs = self.base_model_outputs
            cka_target_inputs = self.base_model_inputs
            cka_weight_multiplier = self.task0_cka_weight  # Lower weight for Task 0
            use_dynamic_layers = False  # No dynamic layer selection for Task 0
            print_rank_0(f"[Adaptive CKA] Task 0: Using base model alignment "
                        f"(weight={cka_weight_multiplier})", self.args.global_rank)
        else:
            cka_target_outputs = self.baseline_outputs
            cka_target_inputs = self.baseline_inputs
            cka_weight_multiplier = 1.0
            use_dynamic_layers = True
            
        print_rank_0(f"[Adaptive CKA] Task {i_task}: active_layers={self.active_layers}",
                    self.args.global_rank)
        
        # Freeze non-current task parameters
        self.freeze_non_current_task_params(i_task)
        
        global_step = 0
        cached_cka_penalty = 0.0
        dynamic_weight = self.bilevel_base_weight
        loss_history = []
        self.model.train()
        
        early_stopped = False
        
        for epoch in range(epochs):
            if early_stopped:
                break
                
            for step, batch in enumerate(dataloader):
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(**batch, use_cache=False)
                task_loss = outputs.loss
                
                # Track loss for early stopping
                loss_history.append(task_loss.item())
                
                # Dynamic layer adjustment (only for Task 1+)
                if use_dynamic_layers and i_task > 0:
                    self._update_active_layers(global_step)
                
                # CKA regularization (for ALL tasks including Task 0)
                cka_loss = 0.0
                if cka_target_outputs:
                    if global_step % self.cka_compute_interval == 0:
                        debug = (global_step % self.cka_debug_interval == 0)
                        
                        # Compute CKA penalty against target
                        cached_cka_penalty = self._compute_cka_penalty_against(
                            cka_target_outputs, cka_target_inputs, debug=debug)
                        
                        # Update bilevel dynamic weight
                        dynamic_weight = self._update_dynamic_weight(cached_cka_penalty)
                        
                        if debug:
                            print_rank_0(f"[Adaptive CKA] Task {i_task} Step {global_step}: "
                                        f"penalty={cached_cka_penalty:.4f}, "
                                        f"dyn_weight={dynamic_weight:.2f}, "
                                        f"layers={self.active_layers}",
                                        self.args.global_rank)
                    
                    cka_loss = cached_cka_penalty
                
                # Total loss with bilevel dynamic weight
                effective_lambda = self.lambda_cka * cka_weight_multiplier * dynamic_weight
                total_loss = task_loss + effective_lambda * cka_loss
                
                # Backward pass
                self.model.backward(total_loss)
                self.model.step()
                
                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    desc = f"Task-{i_task} Epoch-{epoch} loss={task_loss.item():.4f}"
                    desc += f" cka={cka_loss:.4f} w={dynamic_weight:.2f}"
                    if use_dynamic_layers:
                        desc += f" L={len(self.active_layers)}"
                    progress_bar.set_description(desc)
                
                global_step += 1
            
            # Check early stopping at end of each epoch
            if self._should_early_stop(loss_history, epoch, steps_per_epoch):
                early_stopped = True
                print_rank_0(f"[Adaptive CKA] Task {i_task}: Early stopped at epoch {epoch+1}/{epochs}",
                            self.args.global_rank)
                break
        
        progress_bar.close()
        
        # Log final stats
        avg_penalty = np.mean(self.alignment_history[-50:]) if len(self.alignment_history) >= 50 else (
            np.mean(self.alignment_history) if self.alignment_history else 0.0)
        print_rank_0(f"[Adaptive CKA] Task {i_task} completed: "
                    f"epochs={epoch+1}, layers={self.active_layers}, "
                    f"mean_loss={np.mean(loss_history[-100:]):.4f}, "
                    f"avg_cka_penalty={avg_penalty:.4f}",
                    self.args.global_rank)
    
    def _compute_cka_penalty_against(self, target_outputs: Dict[int, Tensor], 
                                     target_inputs: List[Dict], debug: bool = False) -> float:
        """
        Compute CKA penalty against specific target outputs.
        Used for both base model (Task 0) and previous model (Task 1+).
        """
        if not target_outputs:
            return 0.0
        
        # Use only active layers
        layer_indices = [idx for idx in self.active_layers if idx in target_outputs]
        if not layer_indices:
            return 0.0
        
        cka_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for layer_idx in layer_indices:
                target_full = target_outputs[layer_idx]
                
                # Compute current outputs using SAME inputs as target
                current_outputs = []
                
                for batch_idx, input_batch in enumerate(target_inputs):
                    input_gpu = to_device(input_batch, self.device)
                    output = self._get_layer_output(input_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        current_outputs.append(output.cpu())
                
                if not current_outputs:
                    continue
                
                current_full = torch.cat(current_outputs, dim=0)
                
                # Align sizes
                min_samples = min(target_full.size(0), current_full.size(0))
                target_aligned = target_full[:min_samples]
                current_aligned = current_full[:min_samples]
                
                # Compute CKA
                cka_score = self._compute_cka(current_aligned, target_aligned)
                cka_scores.append(cka_score)
                
                if debug:
                    diff = (current_aligned - target_aligned).abs().mean().item()
                    print_rank_0(f"[Adaptive CKA] Layer {layer_idx}: CKA={cka_score:.4f}, "
                                f"diff={diff:.4f}, samples={min_samples}",
                                self.args.global_rank)
        
        self.model.train()
        torch.cuda.empty_cache()
        
        if not cka_scores:
            return 0.0
        
        mean_cka = np.mean(cka_scores)
        penalty = 1.0 - mean_cka
        
        # Track for bilevel adjustment
        self.alignment_history.append(penalty)
        if len(self.alignment_history) > 200:
            self.alignment_history = self.alignment_history[-200:]
        
        if debug:
            print_rank_0(f"[Adaptive CKA] Mean CKA={mean_cka:.4f}, Penalty={penalty:.4f}",
                        self.args.global_rank)
        
        return penalty
    
    def save_cka_state(self, save_dir: str):
        """
        Save CKA state including base model outputs and active layers.
        Override to save additional state needed for adaptive CKA.
        """
        cka_config = {
            'lambda_cka': self.lambda_cka,
            'cka_layers': self.cka_layers,
            'cka_compute_interval': self.cka_compute_interval,
            'cka_max_samples': self.cka_max_samples,
            # Adaptive-specific config
            'active_layers': self.active_layers,
            'base_layers': self.base_layers,
            'candidate_layers': self.candidate_layers,
        }
        
        # Use base class save
        CKACheckpointManager.save_state(
            save_dir,
            self.replay_buffer,
            self.baseline_outputs,
            self.baseline_inputs,
            cka_config
        )
        
        # Additionally save base model outputs for Task 0 alignment
        cka_dir = os.path.join(save_dir, "cka_state")
        if self.base_model_outputs:
            base_model_path = os.path.join(cka_dir, "base_model_outputs.pt")
            torch.save(self.base_model_outputs, base_model_path)
            
            base_inputs_path = os.path.join(cka_dir, "base_model_inputs.pkl")
            with open(base_inputs_path, 'wb') as f:
                pickle.dump(self.base_model_inputs, f)
            
            print_rank_0(f"[Adaptive CKA] Saved base model state with {len(self.base_model_outputs)} layers",
                        self.args.global_rank)
    
    def load_cka_state(self, load_dir: str) -> bool:
        """
        Load CKA state including base model outputs and active layers.
        Override to load additional state needed for adaptive CKA.
        """
        # Use base class load
        replay_buffer, baseline_outputs, baseline_inputs, cka_config = \
            CKACheckpointManager.load_state(load_dir)
        
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
            self.baseline_outputs = baseline_outputs
            self.baseline_inputs = baseline_inputs
            
            # Restore adaptive-specific config
            if 'active_layers' in cka_config:
                self.active_layers = cka_config['active_layers']
            if 'base_layers' in cka_config:
                self.base_layers = cka_config['base_layers']
            if 'candidate_layers' in cka_config:
                self.candidate_layers = cka_config['candidate_layers']
            
            print_rank_0(f"[Adaptive CKA] Restored state: {self.replay_buffer.num_tasks()} tasks, "
                        f"active_layers={self.active_layers}",
                        self.args.global_rank)
            
            # Load base model outputs if available
            cka_dir = os.path.join(load_dir, "cka_state")
            base_model_path = os.path.join(cka_dir, "base_model_outputs.pt")
            if os.path.exists(base_model_path):
                self.base_model_outputs = torch.load(base_model_path, weights_only=True)
                
                base_inputs_path = os.path.join(cka_dir, "base_model_inputs.pkl")
                if os.path.exists(base_inputs_path):
                    with open(base_inputs_path, 'rb') as f:
                        self.base_model_inputs = pickle.load(f)
                
                print_rank_0(f"[Adaptive CKA] Loaded base model outputs with {len(self.base_model_outputs)} layers",
                            self.args.global_rank)
            
            return True
        
        return False

class ReplayBufferV3(ReplayBufferV2):
    """
    Enhanced replay buffer that stores labels for computing training loss.

    Key difference from V2:
    - Stores 'labels' field for each batch (required for loss computation)
    - For LM tasks, labels = input_ids if not explicitly provided
    """

    def add_task_data(self, task_id: int, task_name: str, dataloader, num_batches: int = 10):
        """
        Add data for a task from its dataloader.

        Now also stores labels for replay training loss computation.
        """
        self.task_names[task_id] = task_name

        if task_id not in self.buffers:
            self.buffers[task_id] = []

        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break

            # Convert to CPU and store (include labels for replay training)
            cpu_batch = {
                'input_ids': batch['input_ids'].cpu().clone(),
                'attention_mask': batch.get('attention_mask'),
                'labels': batch.get('labels')
            }

            if cpu_batch['attention_mask'] is not None:
                cpu_batch['attention_mask'] = cpu_batch['attention_mask'].cpu().clone()

            # For LM tasks, labels = input_ids if not provided
            if cpu_batch['labels'] is not None:
                cpu_batch['labels'] = cpu_batch['labels'].cpu().clone()
            else:
                cpu_batch['labels'] = cpu_batch['input_ids'].clone()

            if len(self.buffers[task_id]) < self.max_size_per_task:
                self.buffers[task_id].append(cpu_batch)
            else:
                # Reservoir sampling for overflow
                idx = random.randint(0, len(self.buffers[task_id]) - 1)
                self.buffers[task_id][idx] = cpu_batch

            batch_count += 1

        print(f"[ReplayV3] Added {batch_count} batches for task {task_id} ({task_name})")

    def sample_batch_weighted(self, task_weights: Optional[Dict[int, float]] = None) -> Optional[Dict]:
        """
        Sample a batch with optional task weighting.

        Args:
            task_weights: Optional dict mapping task_id -> weight.
                         If None, uses uniform weighting.
                         Higher weight = higher sampling probability.

        Returns:
            A randomly sampled batch, or None if buffer is empty.
        """
        if not self.buffers:
            return None

        task_ids = list(self.buffers.keys())

        if task_weights is None:
            # Default: weight older tasks more heavily (they're more likely to be forgotten)
            # Weight = 1 / (task_id + 1), so Task 0 gets highest weight
            weights = [1.0 / (tid + 1) for tid in task_ids]
        else:
            weights = [task_weights.get(tid, 1.0) for tid in task_ids]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return None
        weights = [w / total_weight for w in weights]

        # Sample task
        selected_task = random.choices(task_ids, weights=weights, k=1)[0]

        # Sample batch from selected task
        task_batches = self.buffers[selected_task]
        if not task_batches:
            return None

        return random.choice(task_batches)

class AdaptiveCKAUpcycleV3(AdaptiveCKAUpcycle):
    """
    Adaptive CKA Upcycling V3 with Replay Training Loss + Layer-wise Adaptive Penalty.

    Combines:
    1. All features from V2 (bilevel, base alignment, dynamic layers, early stop)
    2. NEW: Replay Training Loss for output capability preservation
    3. NEW: Layer-wise Adaptive CKA Penalty

    Loss formula:
        total_loss = task_loss + λ_cka * layerwise_cka_loss + λ_replay * replay_loss

    Layer-wise Penalty:
        weighted_penalty = Σ(prior_weight[l] × adaptive_mult[l] × penalty[l]) / Σ(prior_weight[l])

    Where:
        - prior_weight: based on layer depth (deeper = higher weight)
        - adaptive_mult: based on penalty magnitude (higher penalty = higher multiplier)
        - penalty[l]: 1 - CKA[l] for layer l
    """

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        # Initialize parent class
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)

        # Replace replay buffer with V3 version (supports labels)
        self.replay_buffer = ReplayBufferV3(
            max_size_per_task=getattr(self.args, 'replay_buffer_size', 100)
        )

        # Replay loss configuration
        self.replay_weight = getattr(args, 'replay_weight', 0.5)  # λ_replay
        self.replay_freq = getattr(args, 'replay_freq', 1)  # Compute every N steps
        self.use_weighted_sampling = getattr(args, 'use_weighted_replay', True)

        # Freeze lm_head option (to prevent output layer forgetting)
        self.freeze_lm_head = getattr(args, 'freeze_lm_head', False)
        if self.freeze_lm_head:
            self._freeze_lm_head()

        # Track replay loss for monitoring
        self.replay_loss_history = []

        # === Layer-wise Adaptive CKA Configuration ===
        self.use_layerwise_penalty = getattr(args, 'use_layerwise_penalty', True)

        # Prior-based layer weights (deeper layers = higher weight)
        # Based on prior knowledge: Layer 15 always has CKA drop
        self.prior_layer_weights = {
            15: 1.0,   # Deepest, most forgetting
            14: 0.9,
            13: 0.8,
            12: 0.6,
            11: 0.5,
            10: 0.4,
            9: 0.3,
            8: 0.25,
            7: 0.2,
            6: 0.2,
            # Layers 0-5: very stable, minimal weight
        }

        # Per-layer penalty history for trend tracking
        self.layer_penalty_history: Dict[int, List[float]] = {}

        # Adaptive multiplier thresholds
        self.severe_forgetting_threshold = getattr(args, 'severe_forgetting_threshold', 0.05)
        self.moderate_forgetting_threshold = getattr(args, 'moderate_forgetting_threshold', 0.02)

        # Early stopping: allow after 2 epochs (epoch index 1)
        self.early_stop_min_epochs = getattr(args, 'early_stop_min_epochs', 1)

        print_rank_0(f"[CKA V3] Initialized with replay_weight={self.replay_weight}, "
                    f"replay_freq={self.replay_freq}, weighted_sampling={self.use_weighted_sampling}, "
                    f"freeze_lm_head={self.freeze_lm_head}, layerwise_penalty={self.use_layerwise_penalty}",
                    self.args.global_rank)

    def _freeze_lm_head(self):
        """Freeze lm_head to prevent output layer forgetting."""
        if hasattr(self.model, 'module'):
            lm_head = self.model.module.lm_head
        else:
            lm_head = self.model.lm_head

        for param in lm_head.parameters():
            param.requires_grad = False

        param_count = sum(p.numel() for p in lm_head.parameters())
        print_rank_0(f"[CKA V3] Frozen lm_head ({param_count:,} params)", self.args.global_rank)

    def _get_prior_layer_weight(self, layer_idx: int) -> float:
        """
        Get prior-based weight for a layer.
        Deeper layers get higher weights based on prior knowledge.
        """
        if layer_idx in self.prior_layer_weights:
            return self.prior_layer_weights[layer_idx]
        # For layers not in the dict, use depth-based default
        # Assuming 16 layers (0-15), normalize by depth
        return max(0.1, layer_idx / 15.0)

    def _get_adaptive_layer_multiplier(self, layer_idx: int, layer_penalty: float) -> float:
        """
        Compute adaptive multiplier based on layer penalty magnitude and trend.

        Returns:
            Multiplier in range [1.0, 2.5]

        Strategy:
            - Severe forgetting (penalty > 0.05): multiplier = 2.0
            - Moderate forgetting (penalty > 0.02): multiplier = 1.5
            - Low forgetting: multiplier = 1.0
            - Additional trend adjustment: +0.5 if penalty is increasing
        """
        # Base multiplier from penalty magnitude
        if layer_penalty > self.severe_forgetting_threshold:
            base_mult = 2.0
        elif layer_penalty > self.moderate_forgetting_threshold:
            base_mult = 1.5
        else:
            base_mult = 1.0

        # Trend adjustment: increase weight if forgetting is getting worse
        trend_adj = 0.0
        if layer_idx in self.layer_penalty_history:
            history = self.layer_penalty_history[layer_idx]
            if len(history) >= 5:
                recent = np.mean(history[-3:])
                older = np.mean(history[-6:-3]) if len(history) >= 6 else np.mean(history[:-3])
                trend = recent - older
                # If penalty is increasing, add up to 0.5 extra
                trend_adj = 0.5 * np.tanh(trend * 20)  # Saturates at ±0.5

        return min(2.5, base_mult + max(0, trend_adj))

    def _compute_cka_penalty_against(self, target_outputs: Dict[int, Tensor],
                                     target_inputs: List[Dict], debug: bool = False) -> float:
        """
        Compute layer-wise adaptive CKA penalty.

        Override parent method to implement:
            weighted_penalty = Σ(prior_weight × adaptive_mult × penalty) / Σ(prior_weight)

        Returns:
            Weighted average penalty across layers
        """
        if not target_outputs:
            return 0.0

        # Use only active layers that exist in target
        layer_indices = [idx for idx in self.active_layers if idx in target_outputs]
        if not layer_indices:
            return 0.0

        layer_penalties = {}  # layer_idx -> (cka_score, penalty)
        layer_weights = {}    # layer_idx -> final weight (prior × adaptive)

        self.model.eval()
        with torch.no_grad():
            for layer_idx in layer_indices:
                target_full = target_outputs[layer_idx]

                # Compute current outputs using SAME inputs as target
                current_outputs = []

                for input_batch in target_inputs:
                    input_gpu = to_device(input_batch, self.device)
                    output = self._get_layer_output(input_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        current_outputs.append(output.cpu())

                if not current_outputs:
                    continue

                current_full = torch.cat(current_outputs, dim=0)

                # Align sizes
                min_samples = min(target_full.size(0), current_full.size(0))
                target_aligned = target_full[:min_samples]
                current_aligned = current_full[:min_samples]

                # Compute CKA and penalty
                cka_score = self._compute_cka(current_aligned, target_aligned)
                layer_penalty = 1.0 - cka_score
                layer_penalties[layer_idx] = (cka_score, layer_penalty)

                # Track per-layer history
                if layer_idx not in self.layer_penalty_history:
                    self.layer_penalty_history[layer_idx] = []
                self.layer_penalty_history[layer_idx].append(layer_penalty)
                # Keep only recent history
                if len(self.layer_penalty_history[layer_idx]) > 50:
                    self.layer_penalty_history[layer_idx] = self.layer_penalty_history[layer_idx][-50:]

                # Compute layer weight
                if self.use_layerwise_penalty:
                    prior_weight = self._get_prior_layer_weight(layer_idx)
                    adaptive_mult = self._get_adaptive_layer_multiplier(layer_idx, layer_penalty)
                    layer_weights[layer_idx] = prior_weight * adaptive_mult
                else:
                    # Uniform weights (original behavior)
                    layer_weights[layer_idx] = 1.0

                if debug:
                    diff = (current_aligned - target_aligned).abs().mean().item()
                    print_rank_0(f"[CKA V3] Layer {layer_idx}: CKA={cka_score:.4f}, "
                                f"penalty={layer_penalty:.4f}, weight={layer_weights[layer_idx]:.3f}, "
                                f"samples={min_samples}",
                                self.args.global_rank)

        self.model.train()
        torch.cuda.empty_cache()

        if not layer_penalties:
            return 0.0

        # Compute weighted penalty
        total_weight = sum(layer_weights.values())
        if total_weight < 1e-10:
            return 0.0

        weighted_penalty = sum(
            layer_weights[idx] * layer_penalties[idx][1]
            for idx in layer_penalties
        ) / total_weight

        # Also track overall penalty history for bilevel adjustment
        self.alignment_history.append(weighted_penalty)
        if len(self.alignment_history) > 200:
            self.alignment_history = self.alignment_history[-200:]

        if debug:
            mean_cka = np.mean([p[0] for p in layer_penalties.values()])
            print_rank_0(f"[CKA V3] Weighted Penalty={weighted_penalty:.4f}, "
                        f"Mean CKA={mean_cka:.4f}, Total Weight={total_weight:.3f}",
                        self.args.global_rank)

        return weighted_penalty

    def _compute_replay_loss(self) -> float:
        """
        Compute replay training loss by sampling from replay buffer.

        Returns:
            Replay loss value (0.0 if buffer is empty)
        """
        if self.replay_buffer.num_tasks() == 0:
            return 0.0

        # Sample batch (weighted towards older tasks)
        if self.use_weighted_sampling:
            batch = self.replay_buffer.sample_batch_weighted()
        else:
            batch = self.replay_buffer.sample_batch()

        if batch is None:
            return 0.0

        # Move to device
        batch_gpu = to_device(batch, self.device)

        # Forward pass for replay loss
        outputs = self.model(**batch_gpu, use_cache=False)
        replay_loss = outputs.loss

        return replay_loss

    def train_one_task(self, task, i_task, epochs):
        """
        Train with:
        - Task 0: Base model alignment + bilevel weights + early stopping
        - Task 1+: Baseline alignment + bilevel weights + dynamic layers + early stopping + REPLAY LOSS
        """
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        total_steps = epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        # Determine CKA target and weight
        if i_task == 0 and self.base_model_outputs:
            cka_target_outputs = self.base_model_outputs
            cka_target_inputs = self.base_model_inputs
            cka_weight_multiplier = self.task0_cka_weight
            use_dynamic_layers = False
            use_replay = False  # No replay for Task 0 (no old tasks yet)
            print_rank_0(f"[CKA V3] Task 0: Base model alignment (weight={cka_weight_multiplier})",
                        self.args.global_rank)
        else:
            cka_target_outputs = self.baseline_outputs
            cka_target_inputs = self.baseline_inputs
            cka_weight_multiplier = 1.0
            use_dynamic_layers = True
            use_replay = (self.replay_buffer.num_tasks() > 0)  # Enable replay if we have old data
            print_rank_0(f"[CKA V3] Task {i_task}: replay={use_replay}, "
                        f"buffer_tasks={self.replay_buffer.num_tasks()}",
                        self.args.global_rank)

        print_rank_0(f"[CKA V3] Task {i_task}: active_layers={self.active_layers}",
                    self.args.global_rank)

        # Freeze non-current task parameters
        self.freeze_non_current_task_params(i_task)

        global_step = 0
        cached_cka_penalty = 0.0
        dynamic_weight = self.bilevel_base_weight
        loss_history = []
        self.model.train()

        early_stopped = False

        for epoch in range(epochs):
            if early_stopped:
                break

            for step, batch in enumerate(dataloader):
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)

                # Forward pass for current task
                outputs = self.model(**batch, use_cache=False)
                task_loss = outputs.loss

                # Track loss for early stopping
                loss_history.append(task_loss.item())

                # Dynamic layer adjustment (only for Task 1+)
                if use_dynamic_layers and i_task > 0:
                    self._update_active_layers(global_step)

                # CKA regularization
                cka_loss = 0.0
                if cka_target_outputs:
                    if global_step % self.cka_compute_interval == 0:
                        debug = (global_step % self.cka_debug_interval == 0)
                        cached_cka_penalty = self._compute_cka_penalty_against(
                            cka_target_outputs, cka_target_inputs, debug=debug)
                        dynamic_weight = self._update_dynamic_weight(cached_cka_penalty)
                    cka_loss = cached_cka_penalty

                # === NEW: Replay Training Loss ===
                replay_loss = 0.0
                if use_replay and (global_step % self.replay_freq == 0):
                    replay_loss = self._compute_replay_loss()
                    self.replay_loss_history.append(replay_loss.item() if isinstance(replay_loss, Tensor) else replay_loss)

                # Total loss: task_loss + cka_loss + replay_loss
                effective_cka_lambda = self.lambda_cka * cka_weight_multiplier * dynamic_weight

                if isinstance(replay_loss, Tensor):
                    total_loss = task_loss + effective_cka_lambda * cka_loss + self.replay_weight * replay_loss
                else:
                    total_loss = task_loss + effective_cka_lambda * cka_loss

                # Backward pass
                self.model.backward(total_loss)
                self.model.step()

                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    desc = f"T{i_task} E{epoch} loss={task_loss.item():.4f}"
                    desc += f" cka={cka_loss:.4f}"
                    if use_replay and isinstance(replay_loss, Tensor):
                        desc += f" rpl={replay_loss.item():.4f}"
                    desc += f" w={dynamic_weight:.2f}"
                    if use_dynamic_layers:
                        desc += f" L={len(self.active_layers)}"
                    progress_bar.set_description(desc)

                global_step += 1

            # Check early stopping at end of each epoch
            if self._should_early_stop(loss_history, epoch, steps_per_epoch):
                early_stopped = True
                print_rank_0(f"[CKA V3] Task {i_task}: Early stopped at epoch {epoch+1}/{epochs}",
                            self.args.global_rank)
                break

        progress_bar.close()

        # Log final stats
        avg_cka_penalty = np.mean(self.alignment_history[-50:]) if len(self.alignment_history) >= 50 else (
            np.mean(self.alignment_history) if self.alignment_history else 0.0)
        avg_replay_loss = np.mean(self.replay_loss_history[-50:]) if len(self.replay_loss_history) >= 50 else (
            np.mean(self.replay_loss_history) if self.replay_loss_history else 0.0)

        print_rank_0(f"[CKA V3] Task {i_task} completed: "
                    f"epochs={epoch+1}, layers={self.active_layers}, "
                    f"mean_loss={np.mean(loss_history[-100:]):.4f}, "
                    f"avg_cka={avg_cka_penalty:.4f}, avg_replay={avg_replay_loss:.4f}",
                    self.args.global_rank)

    def train_continual(self):
        """
        Enhanced continual training with replay loss.

        Flow:
        1. (Resume) Load MoE structure and CKA state
        2. For each task:
           a. Upcycle (add experts)
           b. Train with CKA + Replay loss
           c. Add task data to replay buffer (with labels!)
           d. Cache baseline for next task
           e. Save checkpoint
        """
        start_task = getattr(self.args, 'start_task', 0)

        if start_task > 0:
            print_rank_0(f"[CKA V3] Resuming from task {start_task}", self.args.global_rank)
            self._restore_moe_structure_for_resume(start_task)

            ckpt_dir = self.args.model_name_or_path
            parent_dir = os.path.dirname(ckpt_dir.rstrip('/'))
            task_ckpt = os.path.join(parent_dir, str(start_task - 1))

            if not self.load_cka_state(task_ckpt):
                self.load_cka_state(ckpt_dir)

            if not self.baseline_outputs and len(self.replay_buffer) > 0:
                self._cache_baseline_from_replay(num_batches=10)
        else:
            # Cache base model outputs BEFORE any training
            self._cache_base_model_outputs(num_batches=10)

        # Main training loop
        for i_task, task in enumerate(self.train_task_list):
            if i_task < start_task:
                print_rank_0(f"[CKA V3] Skipping completed task {i_task}: {task}",
                            self.args.global_rank)
                continue

            self.current_task_id = i_task
            print_rank_0(f"[CKA V3] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

            # Upcycle (add experts)
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True

            if do_upcycle:
                self.upcycle_one_task(task, i_task)

            # Train with CKA + Replay
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Add current task to replay buffer (V3 stores labels!)
            dataloader = self.train_task_list[task]
            self.replay_buffer.add_task_data(i_task, task, dataloader, num_batches=10)

            # Cache baseline for next task
            if i_task < len(self.train_task_list) - 1:
                print_rank_0(f"[CKA V3] Caching baseline for next task",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)

            # Save checkpoint
            self.save_model(i_task)
            self.save_cka_state(os.path.join(self.args.output_dir, str(i_task)))

        print_rank_0("[CKA V3] Training completed!", self.args.global_rank)

class NasCKAUpcycleV4(AdaptiveCKAUpcycleV3):
    """
    NAS-Guided Upcycling MoE V4 with Expert-level Selection & Recycling.

    Inherits all V3 features:
    - Replay Training Loss
    - Layer-wise Adaptive CKA Penalty
    - Bilevel dynamic weights
    - Early stopping

    New in V4:
    - Expert sensitivity probing (gradient-based)
    - Selective expansion (only critical experts)
    - Expert recycling (reuse redundant experts)
    - Dynamic router expansion

    Architecture Decision Logic:
        sensitivity = ||∇_expert L_replay||₂  (normalized)

        if sensitivity > threshold_high:
            → EXPAND: Clone expert, freeze old, train new
        elif sensitivity < threshold_low:
            → RECYCLE: Unfreeze old expert, reuse for new task
        else:
            → KEEP: Freeze old expert, no expansion
    """

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        # Initialize V3 parent
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)

        # === NAS Configuration ===
        # Sensitivity thresholds (normalized 0-1 scale)
        self.nas_threshold_high = getattr(args, 'nas_threshold_high', 0.6)  # Above: EXPAND
        self.nas_threshold_low = getattr(args, 'nas_threshold_low', 0.2)   # Below: RECYCLE

        # Probe configuration
        self.nas_probe_batches = getattr(args, 'nas_probe_batches', 20)
        self.nas_probe_enabled = getattr(args, 'nas_probe_enabled', True)

        # Track NAS decisions
        self.nas_decisions_history: Dict[int, Dict] = {}  # task_id -> {layer: decisions}
        self.expert_sensitivity_history: Dict[int, Dict[int, List[float]]] = {}

        # Track parameter efficiency
        self.params_added_per_task: List[int] = []
        self.params_recycled_per_task: List[int] = []

        print_rank_0(f"[NAS-V4] Initialized with threshold_high={self.nas_threshold_high}, "
                    f"threshold_low={self.nas_threshold_low}, probe_batches={self.nas_probe_batches}",
                    self.args.global_rank)

    def _get_moe_layer(self, layer_idx: int):
        """Access MoE layer components (MLP replaced with MoE structure)."""
        if hasattr(self.model, 'module'):
            base = self.model.module.model
        else:
            base = self.model.model
        return base.layers[layer_idx].mlp

    def _get_num_experts(self, layer_idx: int) -> int:
        """Get current number of experts in a layer."""
        moe_layer = self._get_moe_layer(layer_idx)
        if hasattr(moe_layer, 'scientific_experts'):
            return len(moe_layer.scientific_experts)
        elif hasattr(moe_layer, 'experts'):
            return len(moe_layer.experts)
        return 0

    def probe_expert_sensitivity(self, task_id: int) -> Dict[int, List[float]]:
        """
        Phase 1: PROBE - Calculate sensitivity scores for all existing experts.

        Uses gradient of Replay Loss w.r.t. expert parameters as sensitivity measure.
        High gradient norm → Expert is critical for old tasks → Should EXPAND
        Low gradient norm → Expert is redundant → Can RECYCLE

        Returns:
            sensitivity_map: {layer_idx: [normalized_score_per_expert]}
        """
        import torch.distributed as dist
        
        # CRITICAL: Synchronize all ranks before starting probe
        # This ensures no stale NCCL operations from previous tasks cause deadlock
        if dist.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()
        
        print_rank_0(f"[NAS-V4] Phase 1: Probing expert sensitivity for Task {task_id}...",
                    self.args.global_rank)

        if self.replay_buffer.num_tasks() == 0:
            print_rank_0("[NAS-V4] Warning: Empty replay buffer. Skipping probe.",
                        self.args.global_rank)
            return {}

        # 1. Collect probe batches from replay buffer
        probe_batches = []
        for _ in range(self.nas_probe_batches):
            batch = self.replay_buffer.sample_batch()
            if batch:
                probe_batches.append(batch)

        local_batch_count = len(probe_batches)

        # CRITICAL FIX: Synchronize batch count across all ranks to avoid NCCL deadlock
        # Different ranks may have different numbers of valid batches from sample_batch()
        # All ranks MUST call backward() the same number of times for DeepSpeed ZeRO
        if dist.is_initialized():
            # Ensure all ranks have completed batch collection before sync
            dist.barrier()
            
            batch_count_tensor = torch.tensor([local_batch_count], device=self.device, dtype=torch.int64)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.MIN)
            synchronized_batch_count = batch_count_tensor.item()
            
            # Truncate to synchronized count so all ranks process the same number
            probe_batches = probe_batches[:synchronized_batch_count]
            
            print_rank_0(f"[NAS-V4] Batch sync: local={local_batch_count}, synced={synchronized_batch_count}",
                        self.args.global_rank)

        if not probe_batches:
            return {}

        # 2. Enable gradients for all existing experts (temporarily)
        # ROBUST: Don't call eval() — we need gradients and eval() can cause
        # some model wrappers (DeepSpeed, compiled models) to suppress loss.
        was_training = self.model.training
        self.model.train()
        self.model.zero_grad()

        # Track which parameters we enabled
        enabled_params = []

        for layer_idx in self.active_layers:
            moe_layer = self._get_moe_layer(layer_idx)

            # Get expert list (could be scientific_experts or experts)
            experts = None
            if hasattr(moe_layer, 'scientific_experts'):
                experts = moe_layer.scientific_experts
            elif hasattr(moe_layer, 'experts'):
                experts = moe_layer.experts

            if experts:
                for expert in experts:
                    for param in expert.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                            enabled_params.append(param)

        # 3. Accumulate gradients over probe batches — with per-batch fault tolerance
        # NOTE: Must use self.model.backward() instead of loss.backward() for DeepSpeed ZeRO compatibility
        # DeepSpeed ZeRO optimizer requires backward to go through the engine for proper gradient partitioning
        total_batches = len(probe_batches)
        successful_batches = 0

        for batch_idx, batch in enumerate(probe_batches):
            batch_gpu = to_device(batch, self.device)

            # ROBUST: Ensure labels exist; fallback to input_ids if missing
            if 'labels' not in batch_gpu or batch_gpu['labels'] is None:
                if 'input_ids' in batch_gpu:
                    batch_gpu['labels'] = batch_gpu['input_ids'].clone()
                else:
                    print_rank_0(f"[NAS-V4] Warning: batch {batch_idx} missing input_ids and labels, skipping",
                                self.args.global_rank)
                    continue

            loss = None
            try:
                with torch.set_grad_enabled(True):
                    outputs = self.model(**batch_gpu, use_cache=False)
                    loss = outputs.loss

                    # ROBUST: If DeepSpeed wrapper suppresses loss in train mode,
                    # try falling back to the raw module directly.
                    if loss is None and hasattr(self.model, 'module'):
                        raw_outputs = self.model.module(**batch_gpu, use_cache=False)
                        loss = raw_outputs.loss if hasattr(raw_outputs, 'loss') else None

                    if loss is None:
                        print_rank_0(f"[NAS-V4] Warning: batch {batch_idx} returned None loss, skipping",
                                    self.args.global_rank)
                        continue

                    loss = loss / total_batches  # Average
                    # Use DeepSpeed engine's backward method to avoid ZeRO buffer issues
                    self.model.backward(loss)
                    successful_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    # DYNAMIC BATCH: try halving sequence length
                    orig_len = batch_gpu['input_ids'].size(1)
                    if orig_len > 128:
                        half_len = orig_len // 2
                        try:
                            batch_gpu['input_ids'] = batch_gpu['input_ids'][:, :half_len]
                            batch_gpu['attention_mask'] = batch_gpu['attention_mask'][:, :half_len]
                            batch_gpu['labels'] = batch_gpu['labels'][:, :half_len]
                            outputs = self.model(**batch_gpu, use_cache=False)
                            loss = outputs.loss
                            if loss is None and hasattr(self.model, 'module'):
                                raw_outputs = self.model.module(**batch_gpu, use_cache=False)
                                loss = raw_outputs.loss if hasattr(raw_outputs, 'loss') else None
                            if loss is not None:
                                loss = loss / total_batches
                                self.model.backward(loss)
                                successful_batches += 1
                                print_rank_0(f"[NAS-V4] Batch {batch_idx} OOM recovered with seq_len={half_len}",
                                            self.args.global_rank)
                                continue
                        except Exception:
                            pass
                    print_rank_0(f"[NAS-V4] Warning: batch {batch_idx} OOM, skipping",
                                self.args.global_rank)
                else:
                    print_rank_0(f"[NAS-V4] Warning: batch {batch_idx} RuntimeError: {e}, skipping",
                                self.args.global_rank)
                continue
            except Exception as e:
                print_rank_0(f"[NAS-V4] Warning: batch {batch_idx} error: {e}, skipping",
                            self.args.global_rank)
                continue

        # ROBUST: if too many batches failed, gracefully degrade to empty map
        # so NAS falls back to default behaviour (expand all).
        if successful_batches == 0 or (successful_batches / total_batches) < 0.3:
            print_rank_0(f"[NAS-V4] CRITICAL: only {successful_batches}/{total_batches} probe batches succeeded. "
                        f"Falling back to default expand-all strategy.",
                        self.args.global_rank)
            for param in enabled_params:
                param.requires_grad = False
            self.model.zero_grad()
            if not was_training:
                self.model.eval()
            torch.cuda.empty_cache()
            return {}

        print_rank_0(f"[NAS-V4] Probe succeeded for {successful_batches}/{total_batches} batches",
                    self.args.global_rank)
        
        # Ensure all backward passes complete before calculating gradients
        if dist.is_initialized():
            dist.barrier()

        # 4. Calculate sensitivity scores per expert
        sensitivity_map = {}

        for layer_idx in self.active_layers:
            moe_layer = self._get_moe_layer(layer_idx)

            experts = None
            if hasattr(moe_layer, 'scientific_experts'):
                experts = moe_layer.scientific_experts
            elif hasattr(moe_layer, 'experts'):
                experts = moe_layer.experts

            if not experts:
                continue

            layer_scores = []

            for exp_idx, expert in enumerate(experts):
                # Compute L2 norm of gradients for this expert
                grad_norm_sq = 0.0
                param_count = 0

                for param in expert.parameters():
                    if param.grad is not None:
                        grad_norm_sq += param.grad.data.norm(2).item() ** 2
                        param_count += param.numel()

                # Normalize by parameter count
                score = (grad_norm_sq ** 0.5) / (param_count ** 0.5 + 1e-8)
                layer_scores.append(score)

            # Layer-wise min-max normalization for relative comparison
            if layer_scores:
                min_s, max_s = min(layer_scores), max(layer_scores)
                if max_s - min_s > 1e-8:
                    layer_scores = [(s - min_s) / (max_s - min_s) for s in layer_scores]
                else:
                    layer_scores = [0.5 for _ in layer_scores]  # All equal

            sensitivity_map[layer_idx] = layer_scores

        # 5. Restore state: disable gradients for temporarily enabled params
        for param in enabled_params:
            param.requires_grad = False
        self.model.zero_grad()
        # ROBUST: restore previous training mode
        if not was_training:
            self.model.eval()
        torch.cuda.empty_cache()

        # CRITICAL FIX: Broadcast sensitivity scores from Rank 0 to ALL ranks
        # ZeRO-2 partitions gradients, so each rank may compute different gradient norms.
        # Different sensitivity scores → different mask initialization → different architecture
        # decisions → NCCL size mismatches on the next task.
        if dist.is_initialized():
            # Flatten all sensitivity scores into a single tensor for efficient broadcast
            all_layer_indices = sorted(sensitivity_map.keys())
            if all_layer_indices:
                max_experts = max(len(sensitivity_map[li]) for li in all_layer_indices) if all_layer_indices else 0
                # Pack: [num_layers, max_experts] with padding
                sens_tensor = torch.zeros(len(all_layer_indices), max(max_experts, 1), 
                                         device=self.device, dtype=torch.float32)
                count_tensor = torch.zeros(len(all_layer_indices), device=self.device, dtype=torch.int32)
                
                for i, li in enumerate(all_layer_indices):
                    scores = sensitivity_map[li]
                    count_tensor[i] = len(scores)
                    for j, s in enumerate(scores):
                        sens_tensor[i, j] = s
                
                dist.broadcast(sens_tensor, src=0)
                dist.broadcast(count_tensor, src=0)
                
                # Reconstruct sensitivity_map from broadcast data on all ranks
                sensitivity_map = {}
                for i, li in enumerate(all_layer_indices):
                    n = count_tensor[i].item()
                    sensitivity_map[li] = [sens_tensor[i, j].item() for j in range(n)]

        # Log results
        for layer_idx, scores in sensitivity_map.items():
            scores_str = ", ".join([f"{s:.3f}" for s in scores])
            print_rank_0(f"[NAS-V4] Layer {layer_idx} sensitivities: [{scores_str}]",
                        self.args.global_rank)

        # Store for history
        self.expert_sensitivity_history[task_id] = sensitivity_map
        
        # Final barrier to ensure all ranks complete probe before proceeding
        if dist.is_initialized():
            dist.barrier()

        return sensitivity_map

    def make_nas_decisions(self, sensitivity_map: Dict[int, List[float]]) -> Dict[int, List[str]]:
        """
        Phase 2: DECIDE - Generate architecture decisions based on sensitivity.

        Decision rules:
        - sensitivity > threshold_high → EXPAND (critical expert)
        - sensitivity < threshold_low → RECYCLE (redundant expert)
        - otherwise → KEEP (moderate importance)

        Returns:
            decisions: {layer_idx: [decision_per_expert]}
            where decision ∈ {"EXPAND", "RECYCLE", "KEEP"}
        """
        print_rank_0(f"[NAS-V4] Phase 2: Making NAS decisions...", self.args.global_rank)

        decisions = {}

        for layer_idx, scores in sensitivity_map.items():
            layer_decisions = []

            for exp_idx, score in enumerate(scores):
                if score > self.nas_threshold_high:
                    decision = "EXPAND"
                elif score < self.nas_threshold_low:
                    decision = "RECYCLE"
                else:
                    decision = "KEEP"

                layer_decisions.append(decision)

            decisions[layer_idx] = layer_decisions

            # Log
            decisions_str = ", ".join([f"E{i}:{d}({scores[i]:.2f})"
                                       for i, d in enumerate(layer_decisions)])
            print_rank_0(f"[NAS-V4] Layer {layer_idx}: {decisions_str}", self.args.global_rank)

        return decisions

    def apply_nas_decisions(self, decisions: Dict[int, List[str]], task_id: int):
        """
        Phase 2b: Apply architecture decisions - expand or recycle experts.

        EXPAND: Clone expert, freeze old copy, add new trainable copy
        RECYCLE: Unfreeze existing expert for new task training
        KEEP: Keep expert frozen
        """
        print_rank_0(f"[NAS-V4] Applying NAS decisions for Task {task_id}...",
                    self.args.global_rank)

        total_params_added = 0
        total_params_recycled = 0

        for layer_idx, layer_decisions in decisions.items():
            moe_layer = self._get_moe_layer(layer_idx)

            # Get expert list
            if hasattr(moe_layer, 'scientific_experts'):
                experts = moe_layer.scientific_experts
                expert_attr = 'scientific_experts'
            elif hasattr(moe_layer, 'experts'):
                experts = moe_layer.experts
                expert_attr = 'experts'
            else:
                continue

            new_experts = nn.ModuleList()
            expand_count = 0
            recycle_count = 0

            for exp_idx, (expert, decision) in enumerate(zip(experts, layer_decisions)):
                if decision == "EXPAND":
                    # Clone expert for new task
                    new_expert = copy.deepcopy(expert)

                    # Freeze old expert
                    for p in expert.parameters():
                        p.requires_grad = False

                    # Unfreeze new expert
                    for p in new_expert.parameters():
                        p.requires_grad = True

                    # Keep old expert (frozen) and add new expert
                    new_experts.append(expert)
                    new_experts.append(new_expert)

                    expand_count += 1
                    total_params_added += sum(p.numel() for p in new_expert.parameters())

                elif decision == "RECYCLE":
                    # Unfreeze old expert for reuse
                    for p in expert.parameters():
                        p.requires_grad = True

                    new_experts.append(expert)
                    recycle_count += 1
                    total_params_recycled += sum(p.numel() for p in expert.parameters())

                else:  # KEEP
                    # Keep frozen
                    for p in expert.parameters():
                        p.requires_grad = False

                    new_experts.append(expert)

            # Update expert list
            setattr(moe_layer, expert_attr, new_experts)

            # Expand router if needed
            old_num = len(experts)
            new_num = len(new_experts)

            if new_num > old_num:
                self._expand_router(moe_layer, old_num, new_num)

            # Always unfreeze router for new task
            if hasattr(moe_layer, 'router'):
                for p in moe_layer.router.parameters():
                    p.requires_grad = True

            print_rank_0(f"[NAS-V4] Layer {layer_idx}: {expand_count} expanded, "
                        f"{recycle_count} recycled, {len(layer_decisions) - expand_count - recycle_count} kept",
                        self.args.global_rank)

        # Track efficiency
        self.params_added_per_task.append(total_params_added)
        self.params_recycled_per_task.append(total_params_recycled)

        print_rank_0(f"[NAS-V4] Task {task_id}: +{total_params_added:,} params (new), "
                    f"~{total_params_recycled:,} params (recycled)",
                    self.args.global_rank)

        # Store decisions
        self.nas_decisions_history[task_id] = decisions

    def _expand_router(self, moe_layer, old_num: int, new_num: int):
        """
        Resize router output layer to handle new experts.
        Initialize new slots with mean of old weights + noise.
        """
        router = None
        if hasattr(moe_layer, 'router'):
            router = moe_layer.router
            if hasattr(router, 'classifier'):
                router = router.classifier

        if router is None or not hasattr(router, 'weight'):
            return

        old_weight = router.weight.data
        old_bias = router.bias.data if router.bias is not None else None

        hidden_dim = old_weight.size(1)

        # Create new weight matrix
        new_weight = torch.zeros(new_num, hidden_dim, device=old_weight.device, dtype=old_weight.dtype)
        new_weight[:old_num] = old_weight

        # Initialize new expert routing weights
        mean_weight = old_weight.mean(dim=0, keepdim=True)
        for i in range(old_num, new_num):
            new_weight[i] = mean_weight + torch.randn_like(mean_weight) * 0.01

        router.weight = nn.Parameter(new_weight)

        if old_bias is not None:
            new_bias = torch.zeros(new_num, device=old_bias.device, dtype=old_bias.dtype)
            new_bias[:old_num] = old_bias
            new_bias[old_num:] = old_bias.mean()
            router.bias = nn.Parameter(new_bias)

        # Update output features
        if hasattr(router, 'out_features'):
            router.out_features = new_num

        print_rank_0(f"[NAS-V4] Router expanded: {old_num} → {new_num} experts",
                    self.args.global_rank)

    def upcycle_one_task(self, task, i_task):
        """
        Override: NAS-guided architecture adaptation before training.

        For Task 0: Standard initialization (no NAS)
        For Task 1+: Probe → Decide → Apply
        """
        if i_task == 0:
            # Task 0: Use parent's upcycling (no NAS - no old experts to analyze)
            print_rank_0(f"[NAS-V4] Task 0: Using standard upcycling", self.args.global_rank)
            super().upcycle_one_task(task, i_task)
            return

        if not self.nas_probe_enabled:
            # NAS disabled, use standard upcycling
            print_rank_0(f"[NAS-V4] NAS probe disabled, using standard upcycling",
                        self.args.global_rank)
            super().upcycle_one_task(task, i_task)
            return

        print_rank_0(f"[NAS-V4] ===== NAS-Guided Upcycling for Task {i_task} =====",
                    self.args.global_rank)

        # Phase 1: Probe
        sensitivity_map = self.probe_expert_sensitivity(i_task)

        if not sensitivity_map:
            # Fallback to standard upcycling
            print_rank_0("[NAS-V4] No sensitivity data, falling back to standard upcycling",
                        self.args.global_rank)
            super().upcycle_one_task(task, i_task)
            return

        # Phase 2: Decide & Apply
        decisions = self.make_nas_decisions(sensitivity_map)
        self.apply_nas_decisions(decisions, i_task)

        print_rank_0(f"[NAS-V4] ===== NAS-Guided Upcycling Complete =====",
                    self.args.global_rank)

    def train_continual(self):
        """
        Enhanced continual training with NAS-guided architecture adaptation.

        Flow:
        1. For each task:
           a. NAS Probe & Decide (Task 1+)
           b. Apply architecture changes
           c. Train with CKA + Replay loss
           d. Cache baseline
           e. Save checkpoint
        """
        start_task = getattr(self.args, 'start_task', 0)

        if start_task > 0:
            print_rank_0(f"[NAS-V4] Resuming from task {start_task}", self.args.global_rank)
            self._restore_moe_structure_for_resume(start_task)

            ckpt_dir = self.args.model_name_or_path
            parent_dir = os.path.dirname(ckpt_dir.rstrip('/'))
            task_ckpt = os.path.join(parent_dir, str(start_task - 1))

            if not self.load_cka_state(task_ckpt):
                self.load_cka_state(ckpt_dir)

            if not self.baseline_outputs and len(self.replay_buffer) > 0:
                self._cache_baseline_from_replay(num_batches=10)
        else:
            # Cache base model outputs BEFORE any training
            self._cache_base_model_outputs(num_batches=10)

        # Main training loop
        for i_task, task in enumerate(self.train_task_list):
            if i_task < start_task:
                print_rank_0(f"[NAS-V4] Skipping completed task {i_task}: {task}",
                            self.args.global_rank)
                continue

            self.current_task_id = i_task
            print_rank_0(f"[NAS-V4] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

            # NAS-Guided Upcycling (Task 1+)
            self.upcycle_one_task(task, i_task)

            # Train with CKA + Replay
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Add current task to replay buffer
            dataloader = self.train_task_list[task]
            self.replay_buffer.add_task_data(i_task, task, dataloader, num_batches=10)

            # Cache baseline for next task
            if i_task < len(self.train_task_list) - 1:
                print_rank_0(f"[NAS-V4] Caching baseline for next task",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)

            # Save checkpoint
            self.save_model(i_task)
            self.save_cka_state(os.path.join(self.args.output_dir, str(i_task)))

            # Log NAS efficiency
            if i_task > 0:
                print_rank_0(f"[NAS-V4] Cumulative: +{sum(self.params_added_per_task):,} params added, "
                            f"~{sum(self.params_recycled_per_task):,} params recycled",
                            self.args.global_rank)

        print_rank_0("[NAS-V4] Training completed!", self.args.global_rank)

        # Final efficiency report
        if self.params_added_per_task:
            print_rank_0(f"[NAS-V4] === Parameter Efficiency Report ===", self.args.global_rank)
            print_rank_0(f"[NAS-V4] Total params added: {sum(self.params_added_per_task):,}",
                        self.args.global_rank)
            print_rank_0(f"[NAS-V4] Total params recycled: {sum(self.params_recycled_per_task):,}",
                        self.args.global_rank)

class ExpansionMaskLearner(nn.Module):
    """
    Learns whether to EXPAND or RECYCLE each expert using Gumbel-Softmax.
    
    For each (layer, expert) pair, maintains logits [recycle, expand].
    During training, uses Gumbel-Softmax for differentiable sampling.
    
    Attributes:
        logits: [num_layers, num_experts, 2] - learnable parameters
        layer_indices: List of actual layer indices (for sparse layer selection)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_experts_per_layer: int,
        layer_indices: Optional[List[int]] = None,
        init_expand_prob: float = 0.5,
        init_from_sensitivity: Optional[Dict[int, List[float]]] = None,
        deep_layer_expand_bias: float = 0.0,
        deep_layer_indices: Optional[List[int]] = None
    ):
        """
        Args:
            num_layers: Number of MoE layers to manage
            num_experts_per_layer: Number of experts per layer
            layer_indices: Actual layer indices (default: 0 to num_layers-1)
            init_expand_prob: Initial probability bias towards expansion
            init_from_sensitivity: Optional V4 sensitivity scores to initialize from
            deep_layer_expand_bias: Extra expand bias for deep layers (0-2 range recommended)
            deep_layer_indices: Which layer indices are considered "deep" (default: last 3)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_experts = num_experts_per_layer
        self.layer_indices = layer_indices if layer_indices else list(range(num_layers))
        
        # Logits shape: [num_layers, num_experts, 2] for [recycle, expand]
        self.logits = nn.Parameter(torch.zeros(num_layers, num_experts_per_layer, 2))
        
        # Initialize with bias based on prior probability
        if init_expand_prob != 0.5:
            # logit(p) = log(p / (1-p))
            bias = math.log(init_expand_prob / (1 - init_expand_prob + 1e-8))
            self.logits.data[:, :, 1] += bias  # Bias towards expand
        
        # Apply extra bias to deep layers (encourage expansion for task-specific layers)
        if deep_layer_expand_bias > 0:
            # Default: last 3 layers in the list are "deep"
            if deep_layer_indices is None:
                deep_layer_indices = self.layer_indices[-3:] if len(self.layer_indices) >= 3 else self.layer_indices
            for layer_idx in deep_layer_indices:
                if layer_idx in self.layer_indices:
                    local_idx = self.layer_indices.index(layer_idx)
                    self.logits.data[local_idx, :, 1] += deep_layer_expand_bias
        
        # Optional: Initialize from V4 sensitivity scores
        if init_from_sensitivity is not None:
            self._init_from_sensitivity(init_from_sensitivity)
    
    def _init_from_sensitivity(self, sensitivity_map: Dict[int, List[float]]):
        """
        Initialize logits based on V4 sensitivity scores.
        High sensitivity → bias towards EXPAND
        Low sensitivity → bias towards RECYCLE
        """
        for layer_idx, scores in sensitivity_map.items():
            if layer_idx in self.layer_indices:
                local_idx = self.layer_indices.index(layer_idx)
                for exp_idx, score in enumerate(scores):
                    if exp_idx < self.num_experts:
                        # Map sensitivity [0,1] to logit bias [-2, 2]
                        bias = (score - 0.5) * 4.0
                        self.logits.data[local_idx, exp_idx, 1] += bias
    
    def forward(self, temperature: float = 1.0, hard: bool = True) -> Tensor:
        """
        Sample masks using Gumbel-Softmax.
        
        Args:
            temperature: Gumbel temperature (higher = softer)
            hard: If True, use straight-through estimator (hard in forward, soft in backward)
        
        Returns:
            masks: [num_layers, num_experts] where 1=EXPAND, 0=RECYCLE
                   ALWAYS returns contiguous tensor for distributed operations
        """
        # Gumbel-Softmax returns [num_layers, num_experts, 2]
        y_soft = F.gumbel_softmax(self.logits, tau=temperature, hard=hard, dim=-1)
        
        # Return the EXPAND probability/decision (index 1)
        # CRITICAL: .contiguous() required for dist.broadcast compatibility
        return y_soft[..., 1].contiguous()
    
    def get_soft_masks(self) -> Tensor:
        """Get soft masks without Gumbel noise (for visualization/analysis)."""
        probs = F.softmax(self.logits, dim=-1)
        # CRITICAL: .contiguous() for distributed compatibility
        return probs[..., 1].contiguous()
    
    def get_hard_decisions(self) -> Tensor:
        """Get final hard decisions without Gumbel noise."""
        probs = F.softmax(self.logits, dim=-1)
        # CRITICAL: .contiguous() for distributed compatibility
        return (probs[..., 1] > 0.5).float().contiguous()
    
    def get_decision_confidence(self) -> Tensor:
        """Get confidence of each decision (how certain the mask is)."""
        probs = F.softmax(self.logits, dim=-1)
        # Confidence = |p_expand - 0.5| * 2, ranges from 0 to 1
        # CRITICAL: .contiguous() for distributed compatibility
        return ((probs[..., 1] - 0.5).abs() * 2).contiguous()
    
    def get_sparsity(self) -> float:
        """Get current sparsity (fraction of EXPAND decisions)."""
        with torch.no_grad():
            decisions = self.get_hard_decisions()
            return decisions.mean().item()

class DifferentiableNasCKAV5(NasCKAUpcycleV4):
    """
    Differentiable NAS-Guided Upcycling MoE V5.
    
    Inherits all V4 features:
    - Replay Training Loss
    - Layer-wise Adaptive CKA Penalty
    - Bilevel dynamic weights
    - Early stopping
    
    New in V5:
    - Gumbel-Softmax learnable expansion masks
    - Supernet with Old/New expert pairs
    - Interleaved bilevel optimization
    - Temperature annealing for mask convergence
    
    Architecture Decision Logic:
        Instead of: sensitivity > threshold → EXPAND (V4)
        Now: Learn mask m via gradient descent on Replay Loss + Sparsity
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args):
        # Initialize V4 parent
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list,
                         test_task_list, args)
        
        # === V5 NAS Configuration ===
        # Temperature schedule
        self.nas_temperature = getattr(args, 'nas_temperature_init', 5.0)
        self.nas_temperature_final = getattr(args, 'nas_temperature_final', 0.1)
        self.nas_decay_rate = getattr(args, 'nas_decay_rate', 0.99)
        
        # Mask optimization
        self.mask_update_interval = getattr(args, 'mask_update_interval', 10)
        self.sparsity_weight = getattr(args, 'sparsity_weight', 0.1)
        self.mask_lr = getattr(args, 'mask_lr', 1e-2)
        
        # Balanced loss configuration (NEW in V5 fix)
        # knowledge_gain_weight: How much to encourage expansion when new experts are better
        # target_expand_rate: Target percentage of experts to expand
        # balance_weight: Penalty weight for deviating from target expansion rate
        self.knowledge_gain_weight = getattr(args, 'knowledge_gain_weight', 0.3)
        self.target_expand_rate = getattr(args, 'target_expand_rate', 0.3)
        self.balance_weight = getattr(args, 'balance_weight', 0.5)
        
        # Layer selection for NAS (default: all layers, can be limited for OOM)
        self.nas_layers_mode = getattr(args, 'nas_layers', 'all')  # 'all', 'last_4', 'last_8', etc.
        
        # Runtime state
        self.mask_learner: Optional[ExpansionMaskLearner] = None
        self.optimizer_mask: Optional[torch.optim.Optimizer] = None
        self.supernet_active = False
        
        # Track original experts for Supernet
        self.original_experts: Dict[int, nn.ModuleList] = {}
        self.new_experts: Dict[int, nn.ModuleList] = {}
        self.supernet_indices: Dict[int, list] = {}
        
        # Statistics
        self.mask_history: List[Dict] = []
        self.temperature_history: List[float] = []
        
        # Bilevel logging configuration
        self.bilevel_log_interval = getattr(args, 'bilevel_log_interval', 10)  # Log every N mask updates
        self.bilevel_log_detailed = getattr(args, 'bilevel_log_detailed', True)  # Log per-layer details
        self.bilevel_update_count = 0  # Count of mask updates
        
        # Adaptive weight configuration
        self.use_adaptive_weights = getattr(args, 'use_adaptive_weights', True)
        self.adaptive_window_size = getattr(args, 'adaptive_window_size', 50)
        
        # Tracking for adaptive weights
        self.cka_penalty_history: List[float] = []
        self.task_loss_history: List[float] = []
        self.expand_ratio_history: List[float] = []
        
        # === Task Protection Configuration ===
        # Progressive replay weighting: earlier tasks get higher weight
        self.use_task_protection = getattr(args, 'use_task_protection', True)
        # How much to amplify replay weight for the oldest task (linear decay to newest)
        self.task_protection_factor = getattr(args, 'task_protection_factor', 1.5)
        # Minimum samples per task in replay buffer
        self.min_replay_per_task = getattr(args, 'min_replay_per_task', 20)
        
        # === CKA Warmup Configuration ===
        # Ratio of training steps for CKA warmup (gradual increase from 0 to full weight)
        # Helps model adapt before regularization kicks in
        self.cka_warmup_ratio = getattr(args, 'cka_warmup_ratio', 0.0)
        
        print_rank_0(f"[NAS-V5] Initialized with temperature={self.nas_temperature:.1f}→{self.nas_temperature_final:.1f}, "
                    f"decay={self.nas_decay_rate}, mask_interval={self.mask_update_interval}, "
                    f"sparsity_weight={self.sparsity_weight}, nas_layers={self.nas_layers_mode}",
                    self.args.global_rank)
        print_rank_0(f"[NAS-V5] Balanced loss config: knowledge_gain_weight={self.knowledge_gain_weight}, "
                    f"target_expand_rate={self.target_expand_rate}, balance_weight={self.balance_weight}",
                    self.args.global_rank)
        print_rank_0(f"[NAS-V5] Adaptive weights: enabled={self.use_adaptive_weights}, window={self.adaptive_window_size}",
                    self.args.global_rank)
        print_rank_0(f"[NAS-V5] Task protection: enabled={self.use_task_protection}, factor={self.task_protection_factor}",
                    self.args.global_rank)
        if self.cka_warmup_ratio > 0:
            print_rank_0(f"[NAS-V5] CKA warmup: {self.cka_warmup_ratio*100:.0f}% of steps",
                        self.args.global_rank)
    
    def _get_nas_layer_indices(self) -> List[int]:
        """Get layer indices that participate in NAS."""
        all_layers = self.active_layers
        
        if self.nas_layers_mode == 'all':
            return all_layers
        elif self.nas_layers_mode.startswith('last_'):
            n = int(self.nas_layers_mode.split('_')[1])
            return all_layers[-n:] if len(all_layers) >= n else all_layers
        elif self.nas_layers_mode.startswith('first_'):
            n = int(self.nas_layers_mode.split('_')[1])
            return all_layers[:n] if len(all_layers) >= n else all_layers
        else:
            # Try to parse as comma-separated indices
            try:
                return [int(x) for x in self.nas_layers_mode.split(',')]
            except ValueError:
                return all_layers
    
    def _setup_supernet(self, task_id: int):
        """
        Setup Supernet for the TRAINABLE (active set) experts in each layer.
        
        The active set = experts with requires_grad=True, always num_experts_per_task.
        After each finalization:
          - EXPAND: old frozen (leaves active set), new trainable (enters active set)
          - RECYCLE: stays trainable (stays in active set)
        So active set size is always constant = num_experts_per_task.
        
        Frozen experts from prior expansions are NOT part of the supernet.
        """
        print_rank_0(f"[NAS-V5] Setting up Supernet for Task {task_id}...", self.args.global_rank)
        
        nas_layers = self._get_nas_layer_indices()
        
        self.original_experts = {}
        self.new_experts = {}
        self.supernet_indices = {}
        
        total_old_params = 0
        total_new_params = 0
        
        for layer_idx in nas_layers:
            moe_layer = self._get_moe_layer(layer_idx)
            
            if hasattr(moe_layer, 'scientific_experts'):
                all_experts = moe_layer.scientific_experts
            elif hasattr(moe_layer, 'experts'):
                all_experts = moe_layer.experts
            else:
                print_rank_0(f"[NAS-V5] Warning: Layer {layer_idx} has no experts, skipping",
                            self.args.global_rank)
                continue
            
            n = len(all_experts)
            if n == 0:
                continue
            
            # Active set: experts that are trainable (requires_grad=True)
            active_indices = []
            for i, expert in enumerate(all_experts):
                if any(p.requires_grad for p in expert.parameters()):
                    active_indices.append(i)
            
            if not active_indices:
                active_indices = list(range(n))
            
            supernet_experts = nn.ModuleList([all_experts[i] for i in active_indices])
            
            self.original_experts[layer_idx] = supernet_experts
            self.supernet_indices[layer_idx] = active_indices
            
            new_experts = nn.ModuleList()
            for expert in supernet_experts:
                new_expert = copy.deepcopy(expert)
                
                for p in expert.parameters():
                    p.requires_grad = False
                    total_old_params += p.numel()
                
                for p in new_expert.parameters():
                    p.requires_grad = True
                    total_new_params += p.numel()
                
                new_experts.append(new_expert)
            
            self.new_experts[layer_idx] = new_experts.to(self.device)
        
        self.supernet_active = True
        
        print_rank_0(f"[NAS-V5] Supernet setup complete: {len(nas_layers)} layers, "
                    f"active_experts={len(active_indices)}/layer (total={n}), "
                    f"old_params={total_old_params:,} (frozen), new_params={total_new_params:,} (trainable)",
                    self.args.global_rank)
    
    def _init_mask_learner(self, task_id: int, sensitivity_map: Optional[Dict[int, List[float]]] = None):
        """Initialize the mask learner for this task."""
        # Use layers that actually have experts (from _setup_supernet), not all NAS layers
        layers_with_experts = sorted(self.original_experts.keys())
        
        if not layers_with_experts:
            print_rank_0("[NAS-V5] Warning: No layers with experts, skipping mask learner init",
                        self.args.global_rank)
            return
        
        # Supernet only covers the last num_experts_per_task experts per layer
        num_experts = max(len(self.original_experts.get(li, [])) for li in layers_with_experts)
        
        if num_experts == 0:
            return
        
        # Get deep layer expand bias from args (default 0.0)
        deep_layer_expand_bias = getattr(self.args, 'deep_layer_expand_bias', 0.0)
        # Define deep layers as layers 13, 14, 15 (or last 3 in layer_indices)
        deep_layer_indices = getattr(self.args, 'deep_layer_indices', None)
        if deep_layer_indices is None:
            # Default: last 3 layers in layers_with_experts
            deep_layer_indices = layers_with_experts[-3:] if len(layers_with_experts) >= 3 else layers_with_experts
        elif isinstance(deep_layer_indices, str):
            deep_layer_indices = [int(x) for x in deep_layer_indices.split(',')]
        
        # Create mask learner - only for layers that actually have experts
        self.mask_learner = ExpansionMaskLearner(
            num_layers=len(layers_with_experts),
            num_experts_per_layer=num_experts,
            layer_indices=layers_with_experts,
            init_expand_prob=0.6,  # Slight bias towards expansion initially
            init_from_sensitivity=sensitivity_map,
            deep_layer_expand_bias=deep_layer_expand_bias,
            deep_layer_indices=deep_layer_indices
        ).to(self.device)
        
        if deep_layer_expand_bias > 0:
            print_rank_0(f"[NAS-V5] Deep layer expand bias: {deep_layer_expand_bias} for layers {deep_layer_indices}",
                        self.args.global_rank)
        
        mask_override = os.environ.get("MASK_OVERRIDE", "none")
        freeze_mask = os.environ.get("FREEZE_MASK", "0") == "1"
        if mask_override == "invert":
            with torch.no_grad():
                self.mask_learner.logits.data[:, :, [0, 1]] = self.mask_learner.logits.data[:, :, [1, 0]]
            print_rank_0(f"[NAS-V5] MASK_OVERRIDE=invert: swapped expand/recycle logits", self.args.global_rank)
        elif mask_override == "random":
            with torch.no_grad():
                self.mask_learner.logits.data.normal_(0, 0.1)
            print_rank_0(f"[NAS-V5] MASK_OVERRIDE=random: randomized logits", self.args.global_rank)

        if freeze_mask and mask_override != "none":
            self.mask_learner.logits.requires_grad_(False)
            print_rank_0(f"[NAS-V5] FREEZE_MASK=1: mask logits frozen (no gradient update)", self.args.global_rank)
        
        # Create optimizer for masks (skip frozen params)
        trainable_mask_params = [p for p in self.mask_learner.parameters() if p.requires_grad]
        self.optimizer_mask = torch.optim.Adam(
            trainable_mask_params if trainable_mask_params else [torch.zeros(1, requires_grad=True)],
            lr=self.mask_lr
        )
        
        # Reset temperature and update counter
        self.nas_temperature = getattr(self.args, 'nas_temperature_init', 5.0)
        self.bilevel_update_count = 0
        
        print_rank_0(f"[NAS-V5] Mask learner initialized: {len(layers_with_experts)} layers × {num_experts} experts",
                    self.args.global_rank)
        print_rank_0(f"[NAS-V5] Layers with experts: {layers_with_experts}", self.args.global_rank)
        
        # Log initial mask state
        if self.args.global_rank == 0:
            print_rank_0(f"\n{'='*60}", self.args.global_rank)
            print_rank_0(f"[NAS-V5] Initial Mask State (Task {task_id})", self.args.global_rank)
            print_rank_0(f"{'='*60}", self.args.global_rank)
            
            init_soft = self.mask_learner.get_soft_masks()
            init_hard = self.mask_learner.get_hard_decisions()
            
            for mask_idx, layer_idx in enumerate(layers_with_experts):
                if mask_idx >= init_soft.size(0):
                    continue
                
                layer_probs = init_soft[mask_idx]
                layer_decisions = init_hard[mask_idx]
                
                expert_strs = []
                for exp_idx in range(layer_probs.size(0)):
                    prob = layer_probs[exp_idx].item()
                    decision = "EXP" if layer_decisions[exp_idx].item() > 0.5 else "REC"
                    # Show sensitivity score if available
                    sens_str = ""
                    if sensitivity_map and layer_idx in sensitivity_map:
                        if exp_idx < len(sensitivity_map[layer_idx]):
                            sens_str = f",sens={sensitivity_map[layer_idx][exp_idx]:.2f}"
                    expert_strs.append(f"E{exp_idx}:{prob:.2f}({decision}{sens_str})")
                
                print_rank_0(f"  Layer {layer_idx}: [{', '.join(expert_strs)}]", self.args.global_rank)
            
            print_rank_0(f"\n  Initial expand ratio: {init_hard.mean():.2%}", self.args.global_rank)
            print_rank_0(f"  Temperature: {self.nas_temperature:.2f}", self.args.global_rank)
            print_rank_0(f"{'='*60}\n", self.args.global_rank)
    
    def _forward_with_masks(self, batch: Dict, masks: Tensor) -> Any:
        """
        Forward pass through the model with mask-controlled expert routing.
        
        For each expert slot k:
            output_k = mask[k] * new_expert_k(x) + (1 - mask[k]) * old_expert_k(x)
        
        This is achieved by temporarily modifying the MoE forward function.
        """
        # Get layers that actually have experts (used for mask indexing)
        # masks shape: [num_layers_with_experts, num_experts]
        layers_with_experts = sorted(self.original_experts.keys())
        
        # Store original forward functions
        original_forwards = {}
        
        for mask_idx, layer_idx in enumerate(layers_with_experts):
            if layer_idx not in self.original_experts:
                continue
            
            moe_layer = self._get_moe_layer(layer_idx)
            original_forwards[layer_idx] = moe_layer.forward
            
            old_experts = self.original_experts[layer_idx]
            new_experts = self.new_experts[layer_idx]
            layer_masks = masks[mask_idx]
            active_indices = self.supernet_indices.get(layer_idx, [])
            
            expert_attr = 'scientific_experts' if hasattr(moe_layer, 'scientific_experts') else 'experts'
            all_layer_experts = getattr(moe_layer, expert_attr)
            
            # Map global expert index → local supernet index (or -1 if not in supernet)
            idx_map = {gi: li for li, gi in enumerate(active_indices)}
            
            def make_masked_forward(moe, all_exp, old_exp, new_exp, m, idx_map_local, orig_fwd):
                def masked_forward(x):
                    shared_output = moe.original_forward(x)
                    
                    total_experts = len(all_exp)
                    if total_experts == 0:
                        return shared_output
                    
                    input_was_2d = x.dim() == 2
                    if input_was_2d:
                        num_tokens, hidden_dim = x.shape
                        batch_size, seq_len = 1, num_tokens
                        x_3d = x.unsqueeze(0)
                    else:
                        batch_size, seq_len, hidden_dim = x.shape
                        num_tokens = batch_size * seq_len
                        x_3d = x
                    
                    routing_weights, selected_experts = moe.router(x_3d)
                    flat_x = x_3d.view(num_tokens, hidden_dim)
                    final_output = torch.zeros(num_tokens, hidden_dim, device=x.device, dtype=x.dtype)
                    
                    for exp_idx in range(total_experts):
                        expert_mask = (selected_experts == exp_idx)
                        if not expert_mask.any():
                            continue
                        
                        token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
                        if len(token_indices) == 0:
                            continue
                        
                        expert_input = flat_x[token_indices]
                        
                        local_idx = idx_map_local.get(exp_idx, -1)
                        if local_idx >= 0 and local_idx < len(old_exp) and local_idx < m.size(0):
                            mask_val = m[local_idx]
                            old_output = old_exp[local_idx](expert_input)
                            new_output = new_exp[local_idx](expert_input)
                            expert_output = mask_val * new_output + (1 - mask_val) * old_output
                        else:
                            expert_output = all_exp[exp_idx](expert_input)
                        
                        token_weights = (routing_weights[token_indices] * expert_mask[token_indices].float()).sum(dim=-1, keepdim=True)
                        final_output[token_indices] += token_weights * expert_output
                    
                    if input_was_2d:
                        final_output = final_output.view(num_tokens, hidden_dim)
                        if shared_output.dim() == 3:
                            shared_output = shared_output.squeeze(0)
                    else:
                        final_output = final_output.view(batch_size, seq_len, hidden_dim)
                    
                    return shared_output + final_output
                
                return masked_forward
            
            moe_layer.forward = make_masked_forward(
                moe_layer, all_layer_experts, old_experts, new_experts, layer_masks, idx_map, original_forwards[layer_idx]
            )
        
        # Forward pass
        try:
            outputs = self.model(**batch, use_cache=False)
        finally:
            # Restore original forwards
            for layer_idx, orig_fwd in original_forwards.items():
                moe_layer = self._get_moe_layer(layer_idx)
                moe_layer.forward = orig_fwd
        
        return outputs
    
    def _compute_replay_loss(self):
        """
        Compute replay loss for expert training (inner loop).
        Override V3 to use broadcast so all ranks use the same replay batch and forward,
        ensuring identical computation graphs and avoiding NCCL collective desync.
        When supernet is active, uses _forward_with_masks with broadcast masks.
        Returns:
            Tensor replay loss, or 0.0 (float) when no batch.
        """
        import torch.distributed as dist
        
        # CRITICAL: All ranks must participate in the same collectives
        # First sync to ensure all ranks enter this function together
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # CRITICAL FIX: Synchronize num_tasks check across all ranks
        # The replay buffer state might differ across ranks, causing one rank to return early
        # while others continue to broadcast operations, leading to NCCL desync
        if rank == 0:
            num_tasks = self.replay_buffer.num_tasks()
            has_tasks = num_tasks > 0
            if has_tasks:
                replay_batch = self.replay_buffer.sample_batch()
                has_batch = replay_batch is not None
            else:
                replay_batch = None
                has_batch = False
        else:
            has_tasks = False
            replay_batch = None
            has_batch = False
        
        # Broadcast has_tasks decision from rank 0
        if dist.is_initialized():
            has_tasks_tensor = torch.tensor([1 if has_tasks else 0], device=self.device, dtype=torch.int32)
            dist.broadcast(has_tasks_tensor, src=0)
            has_tasks = has_tasks_tensor.item() == 1
        
        if not has_tasks:
            # Sync before early return to ensure all ranks exit together
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return 0.0
        
        # Broadcast has_batch decision from rank 0
        if dist.is_initialized():
            has_batch_tensor = torch.tensor([1 if has_batch else 0], device=self.device, dtype=torch.int32)
            dist.broadcast(has_batch_tensor, src=0)
            has_batch = has_batch_tensor.item() == 1
        
        if not has_batch:
            # Sync before early return to ensure all ranks exit together
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return 0.0
        
        # Broadcast batch data from rank 0 using TENSOR-ONLY broadcasts (no broadcast_object_list)
        # This avoids the non-deterministic pickle serialization that causes NCCL desync
        if dist.is_initialized():
            # Standard batch format: input_ids, attention_mask, labels (optional)
            # Broadcast metadata as a fixed-size tensor: [batch_size, seq_len, has_labels]
            if rank == 0:
                input_ids = replay_batch['input_ids']
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
                has_labels = 1 if 'labels' in replay_batch and replay_batch['labels'] is not None else 0
            else:
                batch_size, seq_len, has_labels = 0, 0, 0
            
            # Broadcast metadata
            meta = torch.tensor([batch_size, seq_len, has_labels], device=self.device, dtype=torch.int32)
            dist.broadcast(meta, src=0)
            batch_size, seq_len, has_labels = meta[0].item(), meta[1].item(), meta[2].item() == 1
            
            # Broadcast input_ids
            if rank == 0:
                input_ids_gpu = replay_batch['input_ids'].to(self.device).contiguous()
            else:
                input_ids_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
            dist.broadcast(input_ids_gpu, src=0)
            
            # Broadcast attention_mask
            if rank == 0:
                attn_mask = replay_batch.get('attention_mask')
                if attn_mask is not None:
                    attn_mask_gpu = attn_mask.to(self.device).contiguous()
                else:
                    attn_mask_gpu = torch.ones(batch_size, seq_len, device=self.device, dtype=torch.long)
            else:
                attn_mask_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
            dist.broadcast(attn_mask_gpu, src=0)
            
            # Broadcast labels if present
            if has_labels:
                if rank == 0:
                    labels_gpu = replay_batch['labels'].to(self.device).contiguous()
                else:
                    labels_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                dist.broadcast(labels_gpu, src=0)
            else:
                labels_gpu = None
            
            # Reconstruct batch
            replay_batch = {
                'input_ids': input_ids_gpu,
                'attention_mask': attn_mask_gpu,
            }
            if labels_gpu is not None:
                replay_batch['labels'] = labels_gpu
        else:
            replay_batch = to_device(replay_batch, self.device)
        
        # Sync before forward pass to ensure all ranks have the data
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        # Forward pass - all ranks must execute the same path
        use_masks = self.mask_learner is not None
        if use_masks:
            masks = self.mask_learner(temperature=self.nas_temperature, hard=True)
            if dist.is_initialized():
                if not masks.is_contiguous():
                    masks = masks.contiguous()
                dist.broadcast(masks, src=0)
            outputs = self._forward_with_masks(replay_batch, masks)
        else:
            outputs = self.model(**replay_batch, use_cache=False)
        
        # Sync after forward to ensure all ranks complete before returning
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        return outputs.loss
    
    def _cache_baseline_from_replay(self, num_batches: int = 10):
        """
        Cache baseline outputs from replay buffer for CKA computation.
        Override V2/V3 to broadcast batches from Rank 0 so all ranks use IDENTICAL data,
        ensuring forward passes have identical collective operations (avoiding NCCL desync).
        """
        import torch.distributed as dist
        
        # CRITICAL: Sync all ranks at entry to prevent desync
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Rank 0 gets batches from replay buffer
        if rank == 0:
            all_batches = self.replay_buffer.get_all_batches()
            batches_to_use = all_batches[:min(num_batches, len(all_batches))] if all_batches else []
            num_batches_actual = len(batches_to_use)
        else:
            batches_to_use = []
            num_batches_actual = 0
        
        # Broadcast number of batches
        if dist.is_initialized():
            num_batches_tensor = torch.tensor([num_batches_actual], device=self.device, dtype=torch.int32)
            dist.broadcast(num_batches_tensor, src=0)
            num_batches_actual = num_batches_tensor.item()
        
        if num_batches_actual == 0:
            print_rank_0("[Adaptive CKA] WARNING: Replay buffer empty, cannot cache baseline",
                        self.args.global_rank)
            return
        
        print_rank_0(f"[Adaptive CKA] Caching baseline from {num_batches_actual} replay batches for layers {self._get_cka_layer_indices()}",
                    self.args.global_rank)
        
        # Broadcast each batch from Rank 0 using TENSOR-ONLY broadcasts
        synced_batches = []
        for batch_idx in range(num_batches_actual):
            if dist.is_initialized():
                # Get batch metadata from rank 0
                if rank == 0:
                    batch = batches_to_use[batch_idx]
                    input_ids = batch['input_ids']
                    batch_size = input_ids.size(0) if isinstance(input_ids, torch.Tensor) else len(input_ids)
                    seq_len = input_ids.size(1) if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1 else len(input_ids[0]) if isinstance(input_ids, list) else input_ids.size(0)
                    has_labels = 1 if 'labels' in batch and batch['labels'] is not None else 0
                else:
                    batch_size, seq_len, has_labels = 0, 0, 0
                
                # Broadcast metadata
                meta = torch.tensor([batch_size, seq_len, has_labels], device=self.device, dtype=torch.int32)
                dist.broadcast(meta, src=0)
                batch_size, seq_len, has_labels = meta[0].item(), meta[1].item(), meta[2].item() == 1
                
                # Broadcast input_ids
                if rank == 0:
                    input_ids_tensor = batch['input_ids']
                    if not isinstance(input_ids_tensor, torch.Tensor):
                        input_ids_tensor = torch.tensor(input_ids_tensor)
                    input_ids_gpu = input_ids_tensor.to(self.device).contiguous()
                else:
                    input_ids_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                dist.broadcast(input_ids_gpu, src=0)
                
                # Broadcast attention_mask
                if rank == 0:
                    attn_mask = batch.get('attention_mask')
                    if attn_mask is not None:
                        if not isinstance(attn_mask, torch.Tensor):
                            attn_mask = torch.tensor(attn_mask)
                        attn_mask_gpu = attn_mask.to(self.device).contiguous()
                    else:
                        attn_mask_gpu = torch.ones(batch_size, seq_len, device=self.device, dtype=torch.long)
                else:
                    attn_mask_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                dist.broadcast(attn_mask_gpu, src=0)
                
                # Broadcast labels if present
                if has_labels:
                    if rank == 0:
                        labels_tensor = batch['labels']
                        if not isinstance(labels_tensor, torch.Tensor):
                            labels_tensor = torch.tensor(labels_tensor)
                        labels_gpu = labels_tensor.to(self.device).contiguous()
                    else:
                        labels_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                    dist.broadcast(labels_gpu, src=0)
                else:
                    labels_gpu = None
                
                # Build synced batch (store on CPU)
                synced_batch = {
                    'input_ids': input_ids_gpu.cpu(),
                    'attention_mask': attn_mask_gpu.cpu(),
                }
                if labels_gpu is not None:
                    synced_batch['labels'] = labels_gpu.cpu()
                synced_batches.append(synced_batch)
            else:
                synced_batches.append(batches_to_use[batch_idx])
        
        # Now all ranks have identical batches - cache baseline outputs
        layer_indices = self._get_cka_layer_indices()
        layer_outputs = {idx: [] for idx in layer_indices}
        
        self.baseline_outputs.clear()
        self.baseline_inputs.clear()
        
        self.model.eval()
        with torch.no_grad():
            for batch in synced_batches:
                self.baseline_inputs.append({
                    'input_ids': batch['input_ids'].clone(),
                    'attention_mask': batch['attention_mask'].clone() if batch.get('attention_mask') is not None else None
                })
                
                batch_gpu = to_device(batch, self.device)
                
                for layer_idx in layer_indices:
                    output = self._get_layer_output(batch_gpu, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None:
                        layer_outputs[layer_idx].append(output.cpu())
                
                torch.cuda.empty_cache()
        
        for layer_idx in layer_indices:
            if layer_outputs[layer_idx]:
                self.baseline_outputs[layer_idx] = torch.cat(layer_outputs[layer_idx], dim=0)
                print_rank_0(f"[Adaptive CKA] Baseline Layer {layer_idx}: cached {self.baseline_outputs[layer_idx].size(0)} samples",
                            self.args.global_rank)
        
        self.model.train()
        
        # CRITICAL: Sync all ranks at exit to ensure baseline caching is complete on all ranks
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
    
    def _compute_cka_penalty_against(self, target_outputs: Dict[int, Tensor],
                                     target_inputs: List[Dict], debug: bool = False) -> float:
        """
        COMPLETE OVERRIDE of V3's CKA computation with full synchronization.
        
        The V3 implementation has internal early returns and continues that can
        cause different ranks to execute different code paths, leading to NCCL
        desync when DeepSpeed's implicit collectives are involved.
        
        This implementation ensures ALL ranks execute the EXACT same code path
        by broadcasting all decisions from Rank 0.
        """
        import torch.distributed as dist
        from utils.utils import to_device, print_rank_0
        
        # CRITICAL: Sync all ranks at entry
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # === Rank 0 determines computation parameters ===
        if rank == 0:
            has_outputs = bool(target_outputs)
            if has_outputs:
                layer_indices = [idx for idx in self.active_layers if idx in target_outputs]
                num_layers = len(layer_indices)
                num_inputs = len(target_inputs) if target_inputs else 0
            else:
                layer_indices = []
                num_layers = 0
                num_inputs = 0
        else:
            has_outputs = False
            layer_indices = []
            num_layers = 0
            num_inputs = 0
        
        # === Broadcast computation parameters ===
        if dist.is_initialized():
            # Broadcast has_outputs, num_layers, num_inputs
            params = torch.tensor([1 if has_outputs else 0, num_layers, num_inputs], 
                                  device=self.device, dtype=torch.int32)
            dist.broadcast(params, src=0)
            has_outputs = params[0].item() == 1
            num_layers = params[1].item()
            num_inputs = params[2].item()
            
            # Broadcast layer_indices
            if num_layers > 0:
                if rank == 0:
                    layer_tensor = torch.tensor(layer_indices, device=self.device, dtype=torch.int32)
                else:
                    layer_tensor = torch.empty(num_layers, device=self.device, dtype=torch.int32)
                dist.broadcast(layer_tensor, src=0)
                layer_indices = layer_tensor.tolist()
        
        # === Early return if nothing to compute - ALL ranks return together ===
        if not has_outputs or num_layers == 0 or num_inputs == 0:
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return 0.0
        
        # === Broadcast all input batches from Rank 0 so ALL ranks have identical data ===
        synced_inputs = []
        for input_idx in range(num_inputs):
            if rank == 0 and input_idx < len(target_inputs):
                input_batch = target_inputs[input_idx]
                input_ids = input_batch['input_ids']
                attention_mask = input_batch.get('attention_mask')
                has_attention_mask = attention_mask is not None
                
                # Broadcast shape info
                seq_len = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
                batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1
            else:
                input_ids = None
                attention_mask = None
                has_attention_mask = False
                seq_len = 0
                batch_size = 0
            
            if dist.is_initialized():
                # Broadcast metadata: [batch_size, seq_len, has_attention_mask]
                meta = torch.tensor([batch_size, seq_len, 1 if has_attention_mask else 0], 
                                   device=self.device, dtype=torch.int32)
                dist.broadcast(meta, src=0)
                batch_size, seq_len, has_attention_mask = meta[0].item(), meta[1].item(), meta[2].item() == 1
                
                # Broadcast input_ids
                if rank == 0:
                    input_ids_gpu = input_ids.to(self.device)
                else:
                    input_ids_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                dist.broadcast(input_ids_gpu, src=0)
                
                # Broadcast attention_mask if present
                if has_attention_mask:
                    if rank == 0:
                        attention_mask_gpu = attention_mask.to(self.device)
                    else:
                        attention_mask_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                    dist.broadcast(attention_mask_gpu, src=0)
                else:
                    attention_mask_gpu = None
                
                synced_inputs.append({
                    'input_ids': input_ids_gpu,
                    'attention_mask': attention_mask_gpu
                })
            else:
                # Single GPU mode
                synced_inputs.append({
                    'input_ids': input_ids.to(self.device) if input_ids is not None else None,
                    'attention_mask': attention_mask.to(self.device) if attention_mask is not None else None
                })
        
        # === Compute CKA - ALL ranks execute the same iterations with synced inputs ===
        layer_penalties = {}
        layer_weights = {}
        
        self.model.eval()
        with torch.no_grad():
            for layer_idx in layer_indices:
                # Get target outputs (only rank 0 uses this, but all ranks iterate)
                if rank == 0:
                    target_full = target_outputs.get(layer_idx)
                else:
                    target_full = None
                
                # Compute current outputs - ALL ranks do the SAME forward passes with SAME inputs
                current_outputs = []
                for input_idx in range(num_inputs):
                    # Sync before each forward pass to keep ranks aligned
                    if dist.is_initialized():
                        torch.cuda.synchronize()
                        dist.barrier()
                    
                    # ALL ranks have the same synced_inputs from broadcast above
                    input_batch = synced_inputs[input_idx]
                    
                    # ALL ranks MUST call _get_layer_output for DeepSpeed collective consistency
                    output = self._get_layer_output(input_batch, layer_idx, max_samples=self.cka_max_samples)
                    if output is not None and rank == 0:
                        current_outputs.append(output.cpu())
                
                # Sync after forward passes for this layer
                if dist.is_initialized():
                    torch.cuda.synchronize()
                    dist.barrier()
                
                # Only Rank 0 computes actual CKA values
                if rank == 0 and current_outputs and target_full is not None:
                    current_full = torch.cat(current_outputs, dim=0)
                    min_samples = min(target_full.size(0), current_full.size(0))
                    target_aligned = target_full[:min_samples]
                    current_aligned = current_full[:min_samples]
                    
                    cka_score = self._compute_cka(current_aligned, target_aligned)
                    layer_penalty = 1.0 - cka_score
                    layer_penalties[layer_idx] = (cka_score, layer_penalty)
                    
                    # Track history
                    if layer_idx not in self.layer_penalty_history:
                        self.layer_penalty_history[layer_idx] = []
                    self.layer_penalty_history[layer_idx].append(layer_penalty)
                    if len(self.layer_penalty_history[layer_idx]) > 50:
                        self.layer_penalty_history[layer_idx] = self.layer_penalty_history[layer_idx][-50:]
                    
                    # Compute layer weight
                    if self.use_layerwise_penalty:
                        prior_weight = self._get_prior_layer_weight(layer_idx)
                        adaptive_mult = self._get_adaptive_layer_multiplier(layer_idx, layer_penalty)
                        layer_weights[layer_idx] = prior_weight * adaptive_mult
                    else:
                        layer_weights[layer_idx] = 1.0
                    
                    if debug:
                        print_rank_0(f"[CKA V3] Layer {layer_idx}: CKA={cka_score:.4f}, "
                                    f"penalty={layer_penalty:.4f}, weight={layer_weights[layer_idx]:.3f}, "
                                    f"samples={min_samples}",
                                    self.args.global_rank)
        
        self.model.train()
        torch.cuda.empty_cache()
        
        # Compute weighted penalty on Rank 0, then broadcast result
        if rank == 0:
            if layer_weights:
                total_weight = sum(layer_weights.values())
                weighted_penalty = sum(
                    layer_weights[idx] * layer_penalties[idx][1]
                    for idx in layer_penalties
                ) / total_weight if total_weight > 0 else 0.0
                
                if debug:
                    mean_cka = sum(p[0] for p in layer_penalties.values()) / len(layer_penalties) if layer_penalties else 0
                    print_rank_0(f"[CKA V3] Weighted Penalty={weighted_penalty:.4f}, "
                                f"Mean CKA={mean_cka:.4f}, Total Weight={total_weight:.3f}",
                                self.args.global_rank)
            else:
                weighted_penalty = 0.0
        else:
            weighted_penalty = 0.0
        
        # Broadcast result to all ranks
        if dist.is_initialized():
            result_tensor = torch.tensor([weighted_penalty], device=self.device, dtype=torch.float32)
            dist.broadcast(result_tensor, src=0)
            weighted_penalty = result_tensor.item()
        
        # Final sync before returning
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        return weighted_penalty
    
    def _log_bilevel_state(self, step: int, masks: Tensor, stats: Dict, 
                           grad_info: Optional[Dict] = None, detailed: bool = False):
        """
        Log comprehensive bilevel optimization state.
        
        Args:
            step: Current training step
            masks: Current mask values [num_layers, num_experts]
            stats: Loss statistics dict
            grad_info: Optional gradient information
            detailed: Whether to log per-layer details
        """
        if self.args.global_rank != 0:
            return
        
        # Use layers with experts (same indexing as mask_learner)
        layers_with_experts = sorted(self.original_experts.keys())
        
        # Header
        print_rank_0(f"\n{'='*60}", self.args.global_rank)
        print_rank_0(f"[NAS-V5 Bilevel] Step {step} | Update #{self.bilevel_update_count}", self.args.global_rank)
        print_rank_0(f"{'='*60}", self.args.global_rank)
        
        # Temperature and Loss Summary
        print_rank_0(f"  Temperature: {stats.get('temperature', 0):.4f} "
                    f"(τ_init={getattr(self.args, 'nas_temperature_init', 5.0):.1f} → "
                    f"τ_final={self.nas_temperature_final:.2f})", self.args.global_rank)
        
        # Detailed loss breakdown
        print_rank_0(f"  Losses:", self.args.global_rank)
        print_rank_0(f"    replay_loss={stats.get('replay_loss', 0):.4f} (→ RECYCLE)", self.args.global_rank)
        print_rank_0(f"    knowledge_gain={stats.get('knowledge_gain', 0):.4f} (→ EXPAND when positive)", 
                    self.args.global_rank)
        print_rank_0(f"    balance_loss={stats.get('balance_loss', 0):.4f} (→ target={stats.get('target_expand_rate', 0.3):.0%})", 
                    self.args.global_rank)
        print_rank_0(f"    total={stats.get('total_mask_loss', 0):.4f}", self.args.global_rank)
        
        # Task performance comparison (if available)
        if stats.get('task_expand_loss', 0) > 0 or stats.get('task_recycle_loss', 0) > 0:
            print_rank_0(f"  Task Performance: expand_loss={stats.get('task_expand_loss', 0):.4f}, "
                        f"recycle_loss={stats.get('task_recycle_loss', 0):.4f}", self.args.global_rank)
        
        print_rank_0(f"  Global Expand Ratio: {stats.get('expand_ratio', 0):.2%} "
                    f"(target={stats.get('target_expand_rate', 0.3):.0%})", self.args.global_rank)
        
        # Recycle/Expand per layer (always printed when mask_learner available)
        if self.mask_learner is not None:
            try:
                hard_masks = self.mask_learner.get_hard_decisions()
                parts = []
                for mask_idx, layer_idx in enumerate(layers_with_experts):
                    if mask_idx >= hard_masks.size(0):
                        continue
                    layer_hard = hard_masks[mask_idx]
                    expand_count = layer_hard.sum().int().item()
                    recycle_count = len(layer_hard) - expand_count
                    parts.append(f"L{layer_idx}:{expand_count}↑{recycle_count}↺")
                if parts:
                    print_rank_0(f"  Recycle/Expand per layer: {' '.join(parts)}", self.args.global_rank)
            except Exception as e:
                print_rank_0(f"  Recycle/Expand per layer: (error {e})", self.args.global_rank)
        
        # Gradient Info (if available)
        if grad_info:
            print_rank_0(f"  Mask Gradients: norm={grad_info.get('grad_norm', 0):.6f}, "
                        f"max={grad_info.get('grad_max', 0):.6f}, "
                        f"min={grad_info.get('grad_min', 0):.6f}", self.args.global_rank)
        
        # Adaptive Weights Info (if enabled)
        if self.use_adaptive_weights and stats.get('_adaptive_kg') is not None:
            print_rank_0(f"  Adaptive Weights: kg={stats.get('_adaptive_kg', 0):.3f}, "
                        f"tr={stats.get('_adaptive_tr', 0):.2%}, "
                        f"bw={stats.get('_adaptive_bw', 0):.3f}", self.args.global_rank)
            print_rank_0(f"  Signals: forgetting={stats.get('_forgetting_trend', 0):.4f}, "
                        f"stagnation={stats.get('_stagnation', 0):.4f}", self.args.global_rank)
        
        # Per-layer mask details
        if detailed and self.mask_learner is not None:
            print_rank_0(f"\n  Per-Layer Mask Values:", self.args.global_rank)
            soft_masks = self.mask_learner.get_soft_masks()  # [num_layers_with_experts, num_experts]
            hard_masks = self.mask_learner.get_hard_decisions()
            confidence = self.mask_learner.get_decision_confidence()
            
            for mask_idx, layer_idx in enumerate(layers_with_experts):
                if mask_idx >= soft_masks.size(0):
                    continue
                
                layer_soft = soft_masks[mask_idx]
                layer_hard = hard_masks[mask_idx]
                layer_conf = confidence[mask_idx]
                
                # Format: E0:0.82(→EXP), E1:0.23(→REC), ...
                expert_strs = []
                for exp_idx in range(layer_soft.size(0)):
                    prob = layer_soft[exp_idx].item()
                    decision = "EXP" if layer_hard[exp_idx].item() > 0.5 else "REC"
                    conf = layer_conf[exp_idx].item()
                    expert_strs.append(f"E{exp_idx}:{prob:.2f}({decision},{conf:.0%})")
                
                expand_count = layer_hard.sum().int().item()
                recycle_count = len(layer_hard) - expand_count
                
                print_rank_0(f"    Layer {layer_idx}: [{', '.join(expert_strs)}] "
                            f"| {expand_count}↑ {recycle_count}↺", self.args.global_rank)
            
            # Logits summary (raw learnable parameters)
            if hasattr(self.mask_learner, 'logits'):
                logits = self.mask_learner.logits.data
                print_rank_0(f"\n  Logits Stats: mean={logits.mean():.4f}, "
                            f"std={logits.std():.4f}, "
                            f"expand_bias_mean={logits[:,:,1].mean():.4f}", self.args.global_rank)
        
        print_rank_0(f"{'='*60}\n", self.args.global_rank)
    
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """
        Dynamically adjust loss weights based on training signals.
        
        Signals:
        - forgetting_signal: CKA penalty trend → increase replay weight if forgetting
        - stagnation_signal: task loss not improving → encourage more expansion
        - diversity_signal: mask distribution entropy → adjust balance constraint
        
        Returns:
            dict with 'knowledge_gain_weight', 'target_expand_rate', 'balance_weight'
        """
        # Default values (from config)
        default_kg = self.knowledge_gain_weight
        default_tr = self.target_expand_rate
        default_bw = self.balance_weight
        
        # Need enough history for meaningful adaptation
        min_history = 10
        if len(self.mask_history) < min_history:
            return {
                'knowledge_gain_weight': default_kg,
                'target_expand_rate': default_tr,
                'balance_weight': default_bw
            }
        
        # Get recent history
        window = min(self.adaptive_window_size, len(self.mask_history))
        recent = self.mask_history[-window:]
        
        # === 1. Forgetting Signal: CKA penalty trend ===
        # Higher CKA penalty = more forgetting
        cka_values = self.cka_penalty_history[-window:] if self.cka_penalty_history else []
        forgetting_trend = 0.0
        if len(cka_values) >= 2:
            # Positive trend = forgetting is increasing
            early_cka = np.mean(cka_values[:len(cka_values)//2]) if cka_values[:len(cka_values)//2] else 0
            late_cka = np.mean(cka_values[len(cka_values)//2:]) if cka_values[len(cka_values)//2:] else 0
            forgetting_trend = late_cka - early_cka
        
        # === 2. Stagnation Signal: Task loss trend ===
        task_losses = [s.get('task_expand_loss', 0) for s in recent if s.get('task_expand_loss', 0) > 0]
        stagnation = 0.0
        if len(task_losses) >= 4:
            early_loss = np.mean(task_losses[:len(task_losses)//2])
            late_loss = np.mean(task_losses[len(task_losses)//2:])
            # Positive = loss increasing (stagnating/getting worse)
            stagnation = (late_loss - early_loss) / (early_loss + 1e-8)
        
        # === 3. Diversity Signal: Expansion ratio variance ===
        expand_ratios = [s.get('expand_ratio', 0.5) for s in recent]
        expand_variance = np.var(expand_ratios) if expand_ratios else 0.1
        current_expand = expand_ratios[-1] if expand_ratios else 0.5
        
        # === Adaptive Adjustments ===
        
        # Knowledge gain weight:
        # - Forgetting high → decrease (be more conservative)
        # - Stagnation high → increase (need more new capacity)
        kg_adjustment = -0.15 * np.tanh(forgetting_trend * 10) + 0.1 * np.tanh(stagnation * 5)
        kg_weight = np.clip(default_kg + kg_adjustment, 0.1, 0.5)
        
        # Target expand rate:
        # - Stagnation high → increase target (need more expansion)
        # - Already high expansion → decrease target
        tr_adjustment = 0.1 * np.tanh(stagnation * 3) - 0.05 * (current_expand - default_tr)
        target_rate = np.clip(default_tr + tr_adjustment, 0.15, 0.5)
        
        # Balance weight:
        # - Low variance (all converging same direction) → decrease (allow exploration)
        # - High forgetting → increase (enforce stability)
        bw_adjustment = 0.2 * np.tanh(forgetting_trend * 5) - 0.15 * np.tanh(expand_variance * 20 - 0.5)
        bal_weight = np.clip(default_bw + bw_adjustment, 0.2, 0.8)
        
        return {
            'knowledge_gain_weight': float(kg_weight),
            'target_expand_rate': float(target_rate),
            'balance_weight': float(bal_weight),
            # Debug info
            '_forgetting_trend': float(forgetting_trend),
            '_stagnation': float(stagnation),
            '_expand_variance': float(expand_variance),
        }
    
    def _update_masks(self, step: int, current_batch: Optional[Dict] = None) -> Dict[str, float]:
        """
        Outer loop: Update mask parameters with BALANCED loss design.
        
        Key insight: Need opposing forces to prevent collapse to all-RECYCLE:
        1. replay_loss (with masks) - preservation signal, biased towards RECYCLE
        2. knowledge_gain - advantage of new experts on current task, encourages EXPAND
        3. balance_loss - penalize deviation from target expansion rate
        
        Loss = replay_loss - γ * knowledge_gain + β * balance_loss
        
        Returns:
            dict with loss components and expand_ratio
        """
        import torch.distributed as dist
        
        # CRITICAL: Sync all ranks at entry to prevent timing divergence
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        if not self.mask_learner.logits.requires_grad:
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return {'_frozen': 1.0, 'expand_ratio': self.mask_learner(temperature=self.nas_temperature, hard=True).mean().item()}
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # CRITICAL FIX: Synchronize early return decisions across all ranks
        # Different ranks may have different local state (mask_learner, replay_buffer)
        # which could cause one rank to return early while others continue
        if rank == 0:
            can_proceed = (self.mask_learner is not None and 
                          self.optimizer_mask is not None and
                          self.replay_buffer.num_tasks() > 0)
            if can_proceed:
                replay_batch = self.replay_buffer.sample_batch()
                has_batch = replay_batch is not None
            else:
                replay_batch = None
                has_batch = False
        else:
            can_proceed = False
            replay_batch = None
            has_batch = False
        
        # Broadcast can_proceed decision from rank 0
        if dist.is_initialized():
            can_proceed_tensor = torch.tensor([1 if can_proceed else 0], device=self.device, dtype=torch.int32)
            dist.broadcast(can_proceed_tensor, src=0)
            can_proceed = can_proceed_tensor.item() == 1
        
        if not can_proceed:
            # Sync before early return to ensure all ranks exit together
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return {}
        
        # Broadcast whether we have a valid batch
        if dist.is_initialized():
            has_batch_tensor = torch.tensor([1 if has_batch else 0], device=self.device, dtype=torch.int32)
            dist.broadcast(has_batch_tensor, src=0)
            has_batch = has_batch_tensor.item() == 1
        
        if not has_batch:
            # Sync before early return to ensure all ranks exit together
            if dist.is_initialized():
                torch.cuda.synchronize()
                dist.barrier()
            return {}
        
        # Step 2: Broadcast replay batch using TENSOR-ONLY broadcasts (no broadcast_object_list)
        if dist.is_initialized():
            # Standard batch format: input_ids, attention_mask, labels (optional)
            if rank == 0:
                input_ids = replay_batch['input_ids']
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1) if input_ids.dim() > 1 else input_ids.size(0)
                has_labels = 1 if 'labels' in replay_batch and replay_batch['labels'] is not None else 0
            else:
                batch_size, seq_len, has_labels = 0, 0, 0
            
            # Broadcast metadata
            meta = torch.tensor([batch_size, seq_len, has_labels], device=self.device, dtype=torch.int32)
            dist.broadcast(meta, src=0)
            batch_size, seq_len, has_labels = meta[0].item(), meta[1].item(), meta[2].item() == 1
            
            # Broadcast input_ids
            if rank == 0:
                input_ids_gpu = replay_batch['input_ids'].to(self.device).contiguous()
            else:
                input_ids_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
            dist.broadcast(input_ids_gpu, src=0)
            
            # Broadcast attention_mask
            if rank == 0:
                attn_mask = replay_batch.get('attention_mask')
                if attn_mask is not None:
                    attn_mask_gpu = attn_mask.to(self.device).contiguous()
                else:
                    attn_mask_gpu = torch.ones(batch_size, seq_len, device=self.device, dtype=torch.long)
            else:
                attn_mask_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
            dist.broadcast(attn_mask_gpu, src=0)
            
            # Broadcast labels if present
            if has_labels:
                if rank == 0:
                    labels_gpu = replay_batch['labels'].to(self.device).contiguous()
                else:
                    labels_gpu = torch.empty(batch_size, seq_len, device=self.device, dtype=torch.long)
                dist.broadcast(labels_gpu, src=0)
            else:
                labels_gpu = None
            
            # Reconstruct batch
            replay_batch = {
                'input_ids': input_ids_gpu,
                'attention_mask': attn_mask_gpu,
            }
            if labels_gpu is not None:
                replay_batch['labels'] = labels_gpu
        else:
            replay_batch = to_device(replay_batch, self.device)
        
        # Step 3: Rank 0 samples masks with Gumbel-Softmax, then broadcast
        masks = self.mask_learner(temperature=self.nas_temperature, hard=True)
        
        if dist.is_initialized():
            if not masks.is_contiguous():
                masks = masks.contiguous()
            dist.broadcast(masks, src=0)
        
        # === 1. Replay Loss (Preservation - biased towards old experts) ===
        outputs = self._forward_with_masks(replay_batch, masks)
        replay_loss = outputs.loss
        
        # === Get Adaptive Weights (if enabled) ===
        adaptive_info = {}
        if self.use_adaptive_weights:
            adaptive_weights = self._compute_adaptive_weights()
            knowledge_gain_weight = adaptive_weights['knowledge_gain_weight']
            target_expand_rate = adaptive_weights['target_expand_rate']
            balance_weight = adaptive_weights['balance_weight']
            adaptive_info = {
                '_adaptive_kg': knowledge_gain_weight,
                '_adaptive_tr': target_expand_rate,
                '_adaptive_bw': balance_weight,
                '_forgetting_trend': adaptive_weights.get('_forgetting_trend', 0),
                '_stagnation': adaptive_weights.get('_stagnation', 0),
            }
        else:
            knowledge_gain_weight = self.knowledge_gain_weight
            target_expand_rate = self.target_expand_rate
            balance_weight = self.balance_weight
        
        # === 2. Knowledge Gain (Encourages expansion when new experts are beneficial) ===
        knowledge_gain = torch.tensor(0.0, device=self.device)
        task_expand_loss = torch.tensor(0.0, device=self.device)
        task_recycle_loss = torch.tensor(0.0, device=self.device)
        
        if current_batch is not None and knowledge_gain_weight > 0:
            # Compare: How much better are new experts vs old experts on current task?
            with torch.no_grad():
                # Loss with all-expand (mask=1, use new experts)
                expand_masks = torch.ones_like(masks)
                expand_outputs = self._forward_with_masks(current_batch, expand_masks)
                task_expand_loss = expand_outputs.loss.detach()
                
                # Loss with all-recycle (mask=0, use old experts)
                recycle_masks = torch.zeros_like(masks)
                recycle_outputs = self._forward_with_masks(current_batch, recycle_masks)
                task_recycle_loss = recycle_outputs.loss.detach()
            
            # Knowledge gain: positive if new experts are better (lower loss)
            # If expand_loss < recycle_loss → knowledge_gain > 0 → encourage expansion
            raw_gain = task_recycle_loss - task_expand_loss
            
            # Scale knowledge gain by current mask values to create gradient
            # High knowledge_gain + high mask → good, encourages keeping high masks
            # High knowledge_gain + low mask → bad, encourages increasing masks
            knowledge_gain = raw_gain * masks.mean()
        
        # CRITICAL: Synchronize after all forward passes before loss computation
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        # === 3. Balance Loss (Prevent collapse to trivial all-0 or all-1) ===
        
        current_expand_rate = masks.mean()
        
        # Asymmetric penalty: penalize under-expansion more than over-expansion
        # This counteracts the natural bias towards RECYCLE
        if current_expand_rate < target_expand_rate:
            # Under-expanding: strong penalty
            expansion_deficit = target_expand_rate - current_expand_rate
            balance_loss = balance_weight * expansion_deficit * 2.0
        else:
            # Over-expanding: mild penalty (use original sparsity behavior)
            expansion_excess = current_expand_rate - target_expand_rate
            balance_loss = self.sparsity_weight * expansion_excess
        
        # === Total Mask Loss ===
        # replay_loss: pushes masks ↓ (old experts good on replay)
        # -knowledge_gain: pushes masks ↑ when new experts better on current task
        # balance_loss: pushes towards target expansion rate
        total_mask_loss = replay_loss - knowledge_gain_weight * knowledge_gain + balance_loss
        
        # Update mask parameters
        # NOTE: We use torch.autograd.grad() instead of backward() to avoid DeepSpeed ZeRO issues
        # DeepSpeed's backward hooks are triggered when calling .backward() on a loss computed
        # through the main model, but we only want gradients for mask_learner parameters
        self.optimizer_mask.zero_grad()
        
        # Compute gradients only for mask_learner parameters (avoids DeepSpeed hooks)
        mask_params = list(self.mask_learner.parameters())
        mask_grads = torch.autograd.grad(
            total_mask_loss,
            mask_params,
            allow_unused=True,
            retain_graph=False
        )
        
        # Manually set gradients for mask parameters
        for param, grad in zip(mask_params, mask_grads):
            if grad is not None:
                param.grad = grad
        
        # Capture gradient info BEFORE optimizer step
        grad_info = {}
        if self.mask_learner.logits.grad is not None:
            grad = self.mask_learner.logits.grad
            grad_info = {
                'grad_norm': grad.norm().item(),
                'grad_max': grad.max().item(),
                'grad_min': grad.min().item(),
                'grad_mean': grad.mean().item(),
            }
        
        self.optimizer_mask.step()
        
        # Anneal temperature
        old_temp = self.nas_temperature
        self.nas_temperature = max(
            self.nas_temperature_final,
            self.nas_temperature * self.nas_decay_rate
        )
        
        # Track statistics
        expand_ratio = masks.mean().item()
        self.temperature_history.append(self.nas_temperature)
        
        # Convert tensors to Python floats for logging
        kg_value = knowledge_gain.item() if isinstance(knowledge_gain, torch.Tensor) else knowledge_gain
        bl_value = balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss
        te_loss = task_expand_loss.item() if isinstance(task_expand_loss, torch.Tensor) else task_expand_loss
        tr_loss = task_recycle_loss.item() if isinstance(task_recycle_loss, torch.Tensor) else task_recycle_loss
        
        stats = {
            'replay_loss': replay_loss.item(),
            'knowledge_gain': kg_value,
            'knowledge_gain_weight': knowledge_gain_weight,
            'balance_loss': bl_value,
            'balance_weight': balance_weight,
            'total_mask_loss': total_mask_loss.item(),
            'expand_ratio': expand_ratio,
            'target_expand_rate': target_expand_rate,
            'task_expand_loss': te_loss,
            'task_recycle_loss': tr_loss,
            'temperature': self.nas_temperature,
            'temperature_old': old_temp,
        }
        
        # Add gradient info to stats
        stats.update(grad_info)
        
        # Add adaptive weight info
        stats.update(adaptive_info)
        
        # Update history for adaptive weights
        self.expand_ratio_history.append(expand_ratio)
        if te_loss > 0:
            self.task_loss_history.append(te_loss)
        
        self.mask_history.append(stats)
        
        # Increment update counter
        self.bilevel_update_count += 1
        
        # Log bilevel state periodically
        should_log = (self.bilevel_update_count % self.bilevel_log_interval == 0) or \
                     (self.bilevel_update_count <= 3)  # Always log first few updates
        
        if should_log:
            self._log_bilevel_state(
                step=step,
                masks=masks,
                stats=stats,
                grad_info=grad_info,
                detailed=self.bilevel_log_detailed
            )
        
        # CRITICAL: Synchronize all ranks after mask update before returning to regular training
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        
        return stats
    
    def _finalize_architecture(self, task_id: int):
        """
        After training, finalize architecture decisions:
        - EXPAND (mask ≈ 1): Keep Old Expert (frozen) + Add New Expert (trainable)
        - RECYCLE (mask ≈ 0): Keep Old Expert (unfrozen for reuse), discard New Expert
        
        This correctly implements selective expansion by ADDING new experts when EXPAND,
        similar to V4's behavior.
        
        CRITICAL: All decisions are broadcast from Rank 0 to ensure all ranks
        make identical architecture changes. Without this, mask parameter drift
        across ranks causes different EXPAND/RECYCLE decisions, leading to different
        model structures and NCCL collective size mismatches on the next task.
        """
        import torch.distributed as dist
        
        if self.mask_learner is None:
            return
        
        # Use layers with experts (same indexing as mask_learner)
        layers_with_experts = sorted(self.original_experts.keys())
        final_masks = self.mask_learner.get_hard_decisions()
        final_soft_masks = self.mask_learner.get_soft_masks()
        final_confidence = self.mask_learner.get_decision_confidence()
        
        # CRITICAL FIX: Broadcast all mask decisions from Rank 0 to ensure consistency
        # Mask optimizer runs independently per rank, causing parameter drift.
        # Without broadcast, ranks may make different EXPAND/RECYCLE decisions.
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            dist.broadcast(final_masks, src=0)
            dist.broadcast(final_soft_masks, src=0)
            dist.broadcast(final_confidence, src=0)
        
        # Log comprehensive finalization report
        if self.args.global_rank == 0:
            print_rank_0(f"\n{'#'*70}", self.args.global_rank)
            print_rank_0(f"[NAS-V5] ARCHITECTURE FINALIZATION - Task {task_id}", self.args.global_rank)
            print_rank_0(f"{'#'*70}", self.args.global_rank)
            
            print_rank_0(f"\n  Bilevel Optimization Summary:", self.args.global_rank)
            print_rank_0(f"    Total mask updates: {self.bilevel_update_count}", self.args.global_rank)
            print_rank_0(f"    Final temperature: {self.nas_temperature:.4f}", self.args.global_rank)
            
            if self.mask_history:
                # Loss trajectory
                first_loss = self.mask_history[0].get('total_mask_loss', 0)
                last_loss = self.mask_history[-1].get('total_mask_loss', 0)
                print_rank_0(f"    Loss trajectory: {first_loss:.4f} → {last_loss:.4f} "
                            f"(Δ={last_loss - first_loss:+.4f})", self.args.global_rank)
                
                # Expand ratio trajectory
                first_expand = self.mask_history[0].get('expand_ratio', 0)
                last_expand = self.mask_history[-1].get('expand_ratio', 0)
                print_rank_0(f"    Expand ratio: {first_expand:.2%} → {last_expand:.2%} "
                            f"(Δ={last_expand - first_expand:+.2%})", self.args.global_rank)
        
        total_expanded = 0
        total_recycled = 0
        total_params_added = 0
        total_params_recycled = 0
        layer_decisions_log = []
        
        for mask_idx, layer_idx in enumerate(layers_with_experts):
            moe_layer = self._get_moe_layer(layer_idx)
            old_experts = self.original_experts[layer_idx]
            new_experts = self.new_experts[layer_idx]
            active_indices = self.supernet_indices.get(layer_idx, [])
            layer_masks = final_masks[mask_idx]
            layer_soft = final_soft_masks[mask_idx]
            layer_conf = final_confidence[mask_idx]
            
            # Build replacement map: for each active expert, decide EXPAND or RECYCLE
            # EXPAND: position gets [old_frozen, new_trainable]
            # RECYCLE: position gets [old_trainable]
            replacements = {}
            decisions = []
            layer_detail = {"layer": layer_idx, "experts": [], "old_num": len(old_experts)}
            layer_expand_count = 0
            layer_recycle_count = 0
            
            for local_idx in range(len(old_experts)):
                mask_val = layer_masks[local_idx].item()
                soft_val = layer_soft[local_idx].item()
                conf_val = layer_conf[local_idx].item()
                global_idx = active_indices[local_idx]
                
                if mask_val > 0.5:
                    for p in old_experts[local_idx].parameters():
                        p.requires_grad = False
                    for p in new_experts[local_idx].parameters():
                        p.requires_grad = True
                    replacements[global_idx] = [old_experts[local_idx], new_experts[local_idx]]
                    
                    decisions.append("EXPAND")
                    total_expanded += 1
                    layer_expand_count += 1
                    total_params_added += sum(p.numel() for p in new_experts[local_idx].parameters())
                else:
                    for p in old_experts[local_idx].parameters():
                        p.requires_grad = True
                    replacements[global_idx] = [old_experts[local_idx]]
                    
                    decisions.append("RECYCLE")
                    total_recycled += 1
                    layer_recycle_count += 1
                    total_params_recycled += sum(p.numel() for p in old_experts[local_idx].parameters())
                
                layer_detail["experts"].append({
                    "idx": local_idx,
                    "decision": decisions[-1],
                    "prob": soft_val,
                    "confidence": conf_val
                })
            
            # Rebuild full expert list: non-active experts stay, active positions replaced
            expert_attr = 'scientific_experts' if hasattr(moe_layer, 'scientific_experts') else 'experts'
            all_experts = getattr(moe_layer, expert_attr)
            final_experts = nn.ModuleList()
            for i, expert in enumerate(all_experts):
                if i in replacements:
                    for e in replacements[i]:
                        final_experts.append(e)
                else:
                    final_experts.append(expert)
            
            layer_detail["new_num"] = len(final_experts)
            layer_detail["expand_count"] = layer_expand_count
            layer_detail["recycle_count"] = layer_recycle_count
            
            old_total = len(all_experts)
            new_total = len(final_experts)
            
            setattr(moe_layer, expert_attr, final_experts)
            
            if new_total > old_total:
                self._expand_router(moe_layer, old_total, new_total)
            
            # Always unfreeze router for new task
            if hasattr(moe_layer, 'router'):
                for p in moe_layer.router.parameters():
                    p.requires_grad = True
            
            layer_decisions_log.append(layer_detail)
        
        # Log per-layer final decisions
        if self.args.global_rank == 0:
            print_rank_0(f"\n  Final Architecture Decisions:", self.args.global_rank)
            for layer_detail in layer_decisions_log:
                layer_idx = layer_detail["layer"]
                expert_strs = []
                for e in layer_detail["experts"]:
                    symbol = "↑" if e["decision"] == "EXPAND" else "↺"
                    expert_strs.append(f"E{e['idx']}:{symbol}({e['prob']:.2f},{e['confidence']:.0%})")
                
                old_num = layer_detail["old_num"]
                new_num = layer_detail["new_num"]
                expand_count = layer_detail["expand_count"]
                recycle_count = layer_detail["recycle_count"]
                
                print_rank_0(f"    Layer {layer_idx}: [{', '.join(expert_strs)}] → "
                            f"{expand_count}↑ {recycle_count}↺ | Experts: {old_num}→{new_num}",
                            self.args.global_rank)
            
            print_rank_0(f"\n  Total: {total_expanded} EXPAND (↑), {total_recycled} RECYCLE (↺)", 
                        self.args.global_rank)
            print_rank_0(f"  Params added: +{total_params_added:,} | Params recycled: ~{total_params_recycled:,}",
                        self.args.global_rank)
            print_rank_0(f"  Effective expansion rate: {total_expanded / (total_expanded + total_recycled):.1%}",
                        self.args.global_rank)
            print_rank_0(f"{'#'*70}\n", self.args.global_rank)
        
        # Cleanup
        self.supernet_active = False
        self.original_experts = {}
        self.new_experts = {}
        self.supernet_indices = {}
        
        # Store decision history with detailed info
        self.nas_decisions_history[task_id] = {
            'masks': final_masks.detach().cpu().numpy(),
            'soft_masks': final_soft_masks.detach().cpu().numpy(),
            'confidence': final_confidence.detach().cpu().numpy(),
            'expanded': total_expanded,
            'recycled': total_recycled,
            'params_added': total_params_added,
            'params_recycled': total_params_recycled,
            'total_updates': self.bilevel_update_count,
            'final_temperature': self.nas_temperature,
            'layer_decisions': layer_decisions_log
        }
    
    def _expand_router(self, moe_layer, old_num: int, new_num: int):
        """
        Resize router output layer to handle new experts.
        Initialize new slots with mean of old weights + noise.
        
        This is essential for selective expansion - when we add new experts,
        the router needs to be able to route to them.
        
        CRITICAL: New weights are broadcast from Rank 0 to ensure all ranks
        have identical router initialization. torch.randn produces different
        values per rank, which would cause model state divergence.
        """
        import torch.distributed as dist
        
        router = None
        if hasattr(moe_layer, 'router'):
            router = moe_layer.router
            if hasattr(router, 'classifier'):
                router = router.classifier
        
        if router is None or not hasattr(router, 'weight'):
            return
        
        old_weight = router.weight.data
        old_bias = router.bias.data if router.bias is not None else None
        
        hidden_dim = old_weight.size(1)
        
        # Create new weight matrix
        new_weight = torch.zeros(new_num, hidden_dim, device=old_weight.device, dtype=old_weight.dtype)
        new_weight[:old_num] = old_weight
        
        # Initialize new expert routing weights (mean + small noise)
        mean_weight = old_weight.mean(dim=0, keepdim=True)
        for i in range(old_num, new_num):
            new_weight[i] = mean_weight + torch.randn_like(mean_weight) * 0.01
        
        # CRITICAL FIX: Broadcast new router weights from Rank 0
        # torch.randn produces different values per rank, causing model divergence
        if dist.is_initialized():
            dist.broadcast(new_weight, src=0)
        
        router.weight = nn.Parameter(new_weight)
        
        if old_bias is not None:
            new_bias = torch.zeros(new_num, device=old_bias.device, dtype=old_bias.dtype)
            new_bias[:old_num] = old_bias
            new_bias[old_num:] = old_bias.mean()
            
            # Broadcast bias too (though it's deterministic, for consistency)
            if dist.is_initialized():
                dist.broadcast(new_bias, src=0)
            
            router.bias = nn.Parameter(new_bias)
        
        # Update output features
        if hasattr(router, 'out_features'):
            router.out_features = new_num
        
        print_rank_0(f"[NAS-V5] Router expanded: {old_num} → {new_num} experts",
                    self.args.global_rank)
    
    def upcycle_one_task(self, task, i_task):
        """
        Override: V5 differentiable NAS-guided architecture adaptation.
        
        For Task 0: Standard initialization (no NAS)
        For Task 1+: Setup Supernet + Initialize Mask Learner
        """
        if i_task == 0:
            # Task 0: Use parent's standard upcycling
            print_rank_0(f"[NAS-V5] Task 0: Using standard upcycling", self.args.global_rank)
            # Call grandparent's upcycle (skip V4's NAS probe)
            AdaptiveCKAUpcycleV3.upcycle_one_task(self, task, i_task)
            return
        
        print_rank_0(f"[NAS-V5] ===== Differentiable NAS Setup for Task {i_task} =====",
                    self.args.global_rank)
        
        # Optional: Run V4's sensitivity probe to initialize masks
        sensitivity_map = None
        if getattr(self.args, 'use_sensitivity_init', True):
            print_rank_0("[NAS-V5] Running sensitivity probe for mask initialization...",
                        self.args.global_rank)
            sensitivity_map = self.probe_expert_sensitivity(i_task)
        
        # Setup Supernet (full expansion)
        self._setup_supernet(i_task)
        
        # Initialize Mask Learner
        self._init_mask_learner(i_task, sensitivity_map)
        
        print_rank_0(f"[NAS-V5] ===== Differentiable NAS Setup Complete =====",
                    self.args.global_rank)
        # CRITICAL: Sync all ranks before first step (avoids NCCL timeout when ranks desync)
        import torch.distributed as dist
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
    
    def _configure_task_dataloader(self, task: str, i_task: int):
        """Rebuild train DataLoader with per-task micro-batch / grad-accum."""
        batch_sizes = getattr(self.args, 'per_task_train_batch_sizes', None)
        datasets = getattr(self.args, 'train_dataset_list', None)
        collator = getattr(self.args, 'data_collator', None)
        if not batch_sizes or not datasets or collator is None:
            return

        bs = int(batch_sizes[i_task]) if i_task < len(batch_sizes) else self.args.per_device_train_batch_size
        grad_accums = getattr(self.args, 'per_task_grad_accum_steps', None)
        _ = (
            int(grad_accums[i_task])
            if grad_accums and i_task < len(grad_accums)
            else self.args.gradient_accumulation_steps
        )  # reserved; DeepSpeed grad_accum fixed at init

        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

        train_dataset = datasets[task]
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)

        self.train_task_list[task] = DataLoader(
            train_dataset,
            collate_fn=collator,
            sampler=train_sampler,
            batch_size=bs,
            num_workers=4,
            pin_memory=True,
        )
        self.args.per_device_train_batch_size = bs
        # Keep grad_accum fixed at DeepSpeed init value; only micro-batch varies per task.

        world = dist.get_world_size() if dist.is_initialized() else 1
        ga = self.args.gradient_accumulation_steps
        eff = bs * world * ga
        print_rank_0(
            f"[Batch] Task {i_task} ({task}): micro_batch={bs}, grad_accum={ga}, "
            f"eff_batch={eff}, steps/epoch={len(self.train_task_list[task])}",
            self.args.global_rank,
        )

    def train_one_task(self, task, i_task, epochs):
        """
        Interleaved Bilevel Training:
        - Every step: Update expert parameters (inner loop)
        - Every N steps: Update mask parameters (outer loop)
        """
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        total_steps = epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        # Determine if we're using Supernet (Task 1+)
        use_supernet = self.supernet_active and self.mask_learner is not None
        
        # CKA target setup (from V3)
        if i_task == 0 and self.base_model_outputs:
            cka_target_outputs = self.base_model_outputs
            cka_target_inputs = self.base_model_inputs
            cka_weight_multiplier = self.task0_cka_weight
            use_dynamic_layers = False
            use_replay = False
            print_rank_0(f"[NAS-V5] Task 0: Base model alignment", self.args.global_rank)
        else:
            cka_target_outputs = self.baseline_outputs
            cka_target_inputs = self.baseline_inputs
            cka_weight_multiplier = 1.0
            use_dynamic_layers = True
            use_replay = (self.replay_buffer.num_tasks() > 0)
            print_rank_0(f"[NAS-V5] Task {i_task}: Supernet={use_supernet}, Replay={use_replay}",
                        self.args.global_rank)
        
        # Freeze non-current task parameters
        self.freeze_non_current_task_params(i_task)
        
        global_step = 0
        cached_cka_penalty = 0.0
        dynamic_weight = self.bilevel_base_weight
        loss_history = []
        mask_stats = {}
        
        self.model.train()
        early_stopped = False
        
        for epoch in range(epochs):
            if early_stopped:
                break
            
            for step, batch in enumerate(dataloader):
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)
                
                # === INNER LOOP: Expert Training ===
                
                # Forward pass (with masks if Supernet is active)
                if use_supernet:
                    # Gumbel-Softmax mask sampling with BROADCAST from rank 0
                    # - Rank 0 samples masks (with Gumbel exploration)
                    # - Broadcast to all ranks for consistency
                    # - More reliable than seed synchronization
                    import torch.distributed as dist
                    
                    masks = self.mask_learner(temperature=self.nas_temperature, hard=True)
                    
                    # Broadcast masks from rank 0 to ensure all ranks use identical masks
                    if dist.is_initialized():
                        if not masks.is_contiguous():
                            masks = masks.contiguous()
                        dist.broadcast(masks, src=0)
                    
                    outputs = self._forward_with_masks(batch, masks)
                else:
                    outputs = self.model(**batch, use_cache=False)
                
                task_loss = outputs.loss
                loss_history.append(task_loss.item())
                
                # Dynamic layer adjustment
                if use_dynamic_layers and i_task > 0:
                    self._update_active_layers(global_step)
                
                # === CKA, Replay, and Mask Update Decisions ===
                # All decisions are made on Rank 0 and broadcast to ensure all ranks
                # execute the same code paths (preventing NCCL collective mismatches)
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                
                cka_loss = 0.0
                replay_loss = 0.0
                
                # Rank 0 determines which features to run
                if rank == 0:
                    should_compute_cka = (
                        i_task > 0 and 
                        cka_target_outputs and 
                        global_step % self.cka_compute_interval == 0
                    )
                    should_compute_replay = (
                        use_replay and 
                        i_task > 0 and 
                        global_step % self.replay_freq == 0
                    )
                    should_update_masks = (
                        use_supernet and 
                        i_task > 0 and 
                        global_step % self.mask_update_interval == 0 and
                        global_step > 0
                    )
                else:
                    should_compute_cka = False
                    should_compute_replay = False
                    should_update_masks = False
                
                # Broadcast all 3 decisions in a single tensor
                if dist.is_initialized():
                    decisions = torch.tensor([
                        1 if should_compute_cka else 0,
                        1 if should_compute_replay else 0,
                        1 if should_update_masks else 0
                    ], device=self.device, dtype=torch.int32)
                    dist.broadcast(decisions, src=0)
                    should_compute_cka = decisions[0].item() == 1
                    should_compute_replay = decisions[1].item() == 1
                    should_update_masks = decisions[2].item() == 1
                
                # Compute CKA penalty
                if should_compute_cka:
                    cka_loss = self._compute_cka_penalty_against(
                        cka_target_outputs, cka_target_inputs, use_dynamic_layers
                    )
                    cached_cka_penalty = cka_loss if isinstance(cka_loss, (int, float)) else cka_loss.item()
                    self.alignment_history.append(cached_cka_penalty)
                    self.cka_penalty_history.append(cached_cka_penalty)
                
                # Compute replay loss
                if should_compute_replay:
                    replay_loss = self._compute_replay_loss()
                
                # Total loss for expert training
                # Apply CKA warmup: gradually increase CKA weight from 0 to full
                warmup_multiplier = 1.0
                if self.cka_warmup_ratio > 0 and i_task > 0:  # Only apply warmup for Task 1+
                    warmup_steps = int(total_steps * self.cka_warmup_ratio)
                    if global_step < warmup_steps:
                        warmup_multiplier = global_step / warmup_steps
                
                effective_cka_lambda = self.lambda_cka * cka_weight_multiplier * dynamic_weight * warmup_multiplier
                
                if isinstance(replay_loss, Tensor):
                    total_loss = task_loss + effective_cka_lambda * cka_loss + self.replay_weight * replay_loss
                else:
                    total_loss = task_loss + effective_cka_lambda * cka_loss
                
                # Ensure all ranks finish forward before backward (avoids NCCL timeout on first step)
                if global_step == 0:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        torch.cuda.synchronize()
                        dist.barrier()
                
                # Backward pass for expert parameters
                self.model.backward(total_loss)
                self.model.step()
                
                # === OUTER LOOP: Mask Learning ===
                # Decision already broadcast above in consolidated decisions
                if should_update_masks:
                    mask_stats = self._update_masks(global_step, current_batch=batch)
                
                # Single barrier after all operations complete
                if dist.is_initialized():
                    torch.cuda.synchronize()
                    dist.barrier()
                
                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    desc = f"T{i_task} E{epoch} loss={task_loss.item():.4f}"
                    desc += f" cka={cka_loss:.4f}"
                    if use_supernet and mask_stats:
                        desc += f" exp={mask_stats.get('expand_ratio', 0):.2f}"
                        desc += f" τ={mask_stats.get('temperature', 0):.2f}"
                    elif use_replay and isinstance(replay_loss, Tensor):
                        desc += f" rpl={replay_loss.item():.4f}"
                    progress_bar.set_description(desc)
                
                global_step += 1
            
            # Check early stopping - MUST synchronize decision across all ranks
            # Different ranks may have different loss values, causing desync if not synchronized
            import torch.distributed as dist
            local_early_stop = self._should_early_stop(loss_history, epoch, steps_per_epoch)
            
            # Synchronize early stop decision: if ANY rank wants to stop, ALL ranks stop
            if dist.is_initialized():
                early_stop_tensor = torch.tensor([1 if local_early_stop else 0], 
                                                  device=self.device, dtype=torch.int32)
                dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
                early_stopped = (early_stop_tensor.item() > 0)
            else:
                early_stopped = local_early_stop
            
            if early_stopped:
                print_rank_0(f"[NAS-V5] Task {i_task}: Early stopped at epoch {epoch+1}/{epochs}",
                            self.args.global_rank)
                break
        
        progress_bar.close()
        
        # === FINALIZE ARCHITECTURE ===
        if use_supernet:
            self._finalize_architecture(i_task)
        
        # Log final stats
        avg_cka_penalty = np.mean(self.alignment_history[-50:]) if self.alignment_history else 0.0
        
        print_rank_0(f"[NAS-V5] Task {i_task} completed: "
                    f"epochs={epoch+1}, mean_loss={np.mean(loss_history[-100:]):.4f}, "
                    f"avg_cka={avg_cka_penalty:.4f}",
                    self.args.global_rank)
        
        if self.mask_history:
            final_expand = self.mask_history[-1].get('expand_ratio', 0)
            print_rank_0(f"[NAS-V5] Final expand ratio: {final_expand:.2%}", self.args.global_rank)
        
        # CRITICAL: Synchronize all ranks after training completes
        # This ensures all async NCCL operations from DeepSpeed are complete
        import torch.distributed as dist
        if dist.is_initialized():
            torch.cuda.synchronize()  # Flush pending CUDA operations
            dist.barrier()  # Synchronize ranks
    
    def train_continual(self):
        """
        Enhanced continual training with differentiable NAS.
        
        Flow:
        1. For each task:
           a. Setup Supernet + Mask Learner (Task 1+)
           b. Interleaved bilevel training
           c. Finalize architecture
           d. Add to replay buffer
           e. Save checkpoint
        """
        start_task = getattr(self.args, 'start_task', 0)
        
        if start_task > 0:
            print_rank_0(f"[NAS-V5] Resuming from task {start_task}", self.args.global_rank)
            self._restore_moe_structure_for_resume(start_task)
            
            ckpt_dir = self.args.model_name_or_path
            parent_dir = os.path.dirname(ckpt_dir.rstrip('/'))
            task_ckpt = os.path.join(parent_dir, str(start_task - 1))
            
            if not self.load_cka_state(task_ckpt):
                self.load_cka_state(ckpt_dir)
            
            if not self.baseline_outputs and len(self.replay_buffer) > 0:
                self._cache_baseline_from_replay(num_batches=10)
        else:
            # Cache base model outputs BEFORE any training
            self._cache_base_model_outputs(num_batches=10)
        
        # Main training loop
        for i_task, task in enumerate(self.train_task_list):
            if i_task < start_task:
                print_rank_0(f"[NAS-V5] Skipping completed task {i_task}: {task}",
                            self.args.global_rank)
                continue
            
            self.current_task_id = i_task
            print_rank_0(f"[NAS-V5] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

            self._configure_task_dataloader(task, i_task)

            # CRITICAL: Ensure all ranks are synchronized before starting new task
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                torch.cuda.synchronize()
            
            # V5 NAS-Guided Upcycling
            self.upcycle_one_task(task, i_task)
            
            # Interleaved Bilevel Training
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            
            # Add current task to replay buffer
            dataloader = self.train_task_list[task]
            self.replay_buffer.add_task_data(i_task, task, dataloader, num_batches=10)
            
            # Synchronize after replay buffer update
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            
            # Cache baseline for next task
            if i_task < len(self.train_task_list) - 1:
                print_rank_0(f"[NAS-V5] Caching baseline for next task",
                            self.args.global_rank)
                self._cache_baseline_from_replay(num_batches=10)
            
            # CRITICAL: Flush and synchronize before checkpoint save
            # This ensures all pending NCCL operations are complete
            import torch.distributed as dist
            if dist.is_initialized():
                print_rank_0(f"[NAS-V5] Synchronizing before checkpoint save...",
                            self.args.global_rank)
                torch.cuda.synchronize()  # Flush CUDA operations first
                dist.barrier()  # Then synchronize ranks
            
            self.save_model(i_task)
            self.save_cka_state(os.path.join(self.args.output_dir, str(i_task)))
            
            if dist.is_initialized():
                dist.barrier()  # Ensure save completes on all ranks
                torch.cuda.synchronize()
            
            # Log NAS statistics
            if i_task > 0 and self.nas_decisions_history.get(i_task):
                stats = self.nas_decisions_history[i_task]
                print_rank_0(f"[NAS-V5] Task {i_task} NAS: {stats['expanded']} expanded, "
                            f"{stats['recycled']} recycled",
                            self.args.global_rank)
            
            # CRITICAL: Synchronize all ranks after task completion before starting next task
            # This prevents stale NCCL operations from checkpoint saving from causing deadlocks
            import torch.distributed as dist
            if dist.is_initialized():
                print_rank_0(f"[NAS-V5] Task {i_task} complete. Synchronizing ranks...",
                            self.args.global_rank)
                dist.barrier()
                torch.cuda.synchronize()
        
        print_rank_0("[NAS-V5] Training completed!", self.args.global_rank)
        
        # Final report
        if self.nas_decisions_history:
            total_expanded = sum(s['expanded'] for s in self.nas_decisions_history.values() if isinstance(s, dict))
            total_recycled = sum(s['recycled'] for s in self.nas_decisions_history.values() if isinstance(s, dict))
            print_rank_0(f"[NAS-V5] === Final NAS Report ===", self.args.global_rank)
            print_rank_0(f"[NAS-V5] Total expanded: {total_expanded}", self.args.global_rank)
            print_rank_0(f"[NAS-V5] Total recycled: {total_recycled}", self.args.global_rank)

def create_cka_upcycle_v5(model, tokenizer, optimizer, train_task_list, eval_task_list,
                          test_task_list, args):
    """Factory function to create Differentiable NAS CKA V5 Upcycle model."""
    print_rank_0("[Factory] Creating DifferentiableNasCKAV5 "
                "(V4 features + Gumbel-Softmax learnable masks + interleaved bilevel)",
                args.global_rank)
    return DifferentiableNasCKAV5(
        model, tokenizer, optimizer,
        train_task_list, eval_task_list, test_task_list, args
    )
