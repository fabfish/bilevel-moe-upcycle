"""
Upcycling MoE Implementation for Prototype
Simplified and refactored version for continual learning.

Supports two initialization modes for Task 0:
- Non-expansion mode: Dense -> 0~7 experts (all trainable)
- Expansion mode: Dense -> 0~7 (frozen) -> expand to 8~15 (trainable)

Evaluation Framework:
=====================
- CKA: Evaluates on OLD task data to measure knowledge retention
  - Task 0: Compare MoE vs Dense (or 8~15 vs 0~7 in expansion mode) on Task 0 data
  - Task N: Compare pre-expansion vs post-expansion output on data from Task 0~N-1
  
- Flatness: Evaluates on NEW task data to measure trainability
  - Task 0: Evaluate Dense FFN (or experts 0~7) flatness on Task 0 data
  - Task N: Evaluate old experts flatness on Task N data (guides expansion decision)

This design supports:
1. Knowledge retention measurement (CKA on old tasks)
2. Expansion decision making (Flatness on new task)
3. Dynamic expansion based on evaluation results
"""

from copy import deepcopy
from typing import Optional, Callable, List, Dict, Any
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from utils.utils import print_rank_0, to_device
from transformers import GenerationConfig
import json
import os
import types
import time

# Import fast flatness evaluator
try:
    from evaluations.fast_flatness import (
        FastFlatnessEvaluator,
        FlatnessVisualizer,
        evaluate_all_experts_fast,
        create_reconstruction_loss_fn
    )
    FAST_FLATNESS_AVAILABLE = True
except ImportError:
    FAST_FLATNESS_AVAILABLE = False

# Task evaluators (imported lazily when needed)
_TASK_EVALUATORS = None

def _get_task_evaluators():
    global _TASK_EVALUATORS
    if _TASK_EVALUATORS is None:
        from evaluations import (
            eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, 
            eval_CStance, eval_Py150, eval_FOMC, 
            eval_NumGLUE_cm, eval_NumGLUE_ds
        )
        _TASK_EVALUATORS = {
            "ScienceQA": eval_ScienceQA.eval,
            "MeetingBank": eval_MeetingBank.eval,
            "C-STANCE": eval_CStance.eval,
            "Papyrus-f": eval_PapyrusF.eval,
            "Py150": eval_Py150.eval,
            "FOMC": eval_FOMC.eval,
            "NumGLUE-cm": eval_NumGLUE_cm.eval,
            "NumGLUE-ds": eval_NumGLUE_ds.eval,
        }
    return _TASK_EVALUATORS

generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


# =============================================================================
# MoE Core Components
# =============================================================================

class Expert(nn.Module):
    """A standard MLP expert."""
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Router module for dispatching tokens to experts."""
    def __init__(self, config, num_experts):
        super().__init__()
        self.top_k = config.num_activated_experts
        self.classifier = nn.Linear(config.hidden_size, num_experts)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        logits = self.classifier(hidden_states)
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


def moe_forward(self, x):
    """
    Optimized MoE forward pass using batched expert computation.
    This function will be monkey-patched into each LlamaMLP instance.
    """
    # 1. Shared Expert (Original FFN)
    shared_expert_output = self.original_forward(x)

    # 2. Scientific Experts (batched implementation)
    if len(self.scientific_experts) > 0:
        input_was_2d = x.dim() == 2
        if input_was_2d:
            num_tokens, hidden_dim = x.shape
            batch_size, seq_len = 1, num_tokens
            x_3d = x.unsqueeze(0)
        else:
            batch_size, seq_len, hidden_dim = x.shape
            num_tokens = batch_size * seq_len
            x_3d = x
        
        routing_weights, selected_experts = self.router(x_3d)
        flat_x = x_3d.view(num_tokens, hidden_dim)
        final_expert_output = torch.zeros(num_tokens, hidden_dim, device=x.device, dtype=x.dtype)
        
        num_experts = len(self.scientific_experts)
        for expert_idx in range(num_experts):
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue
            
            expert_input = flat_x[token_indices]
            expert_output = self.scientific_experts[expert_idx](expert_input)
            token_weights = (routing_weights[token_indices] * expert_mask[token_indices].float()).sum(dim=-1, keepdim=True)
            final_expert_output[token_indices] += token_weights * expert_output
        
        if input_was_2d:
            final_expert_output = final_expert_output.view(num_tokens, hidden_dim)
            if shared_expert_output.dim() == 3:
                shared_expert_output = shared_expert_output.squeeze(0)
        else:
            final_expert_output = final_expert_output.view(batch_size, seq_len, hidden_dim)
        
        return shared_expert_output + final_expert_output
    else:
        return shared_expert_output


# =============================================================================
# Model Conversion Function (for inference)
# =============================================================================

def convert_upcycle_model(model, args, num_tasks=0, incremental=False):
    """
    Convert the model to an Upcycling MoE model for inference.
    
    This function is from upcycling_refactored.py (not upcycling.py).

    - If incremental == False: build model with num_tasks * num_experts_per_task experts.
    - If incremental == True: append num_tasks * num_experts_per_task new experts to existing MoE.
    
    Args:
        model: The base model to convert
        args: Arguments containing num_experts_per_task, num_activated_experts, local_rank
        num_tasks: Number of tasks (determines total expert count)
        incremental: Whether to add experts incrementally or build from scratch
    
    Returns:
        model: The converted MoE model
    """
    num_experts = int(getattr(args, 'num_experts_per_task', 8))
    local_rank = getattr(args, 'local_rank', 0)
    device = torch.device("cuda")
    model_dtype = next(model.parameters()).dtype
    
    # Log conversion info
    print_rank_0(f"[Upcycle-Refactored] Using module: upcycling_refactored.py", local_rank)
    print_rank_0(f"[Upcycle-Refactored] Conversion params: num_tasks={num_tasks}, incremental={incremental}, num_experts_per_task={num_experts}", local_rank)

    if incremental:
        num_new_experts = int(num_tasks) * num_experts
        if num_new_experts <= 0:
            print_rank_0(f"[Upcycle-Refactored] incremental=True but num_new_experts=={num_new_experts}, nothing to do.", local_rank)
            return model
        print_rank_0(f"[Upcycle-Refactored] ⚙️ Incrementally adding {num_new_experts} experts per layer.", local_rank)

        for li, layer in enumerate(model.model.layers):
            if li % 2 != 0:
                pass

            mlp = layer.mlp
            if not hasattr(mlp, "scientific_experts") or len(mlp.scientific_experts) == 0:
                mlp.original_forward = mlp.forward
                mlp.scientific_experts = nn.ModuleList([])
                if not hasattr(model.model.config, 'num_activated_experts'):
                    model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))
                mlp.router = Router(model.model.config, 0).to(device=device, dtype=model_dtype)

            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data

            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            new_intermediate_size = H // num_experts

            old_total = len(mlp.scientific_experts)
            for idx in range(num_new_experts):
                new_expert = Expert(model.model.config, new_intermediate_size).to(device=device, dtype=model_dtype)
                expert_idx_in_task = (old_total + idx) % num_experts
                start_col = expert_idx_in_task * new_intermediate_size
                end_col = (expert_idx_in_task + 1) * new_intermediate_size

                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

                mlp.scientific_experts.append(new_expert)

            old_classifier = mlp.router.classifier
            old_out = old_classifier.weight.data.shape[0] if hasattr(old_classifier, 'weight') and old_classifier.weight is not None else 0
            old_in = h
            new_out = old_out + num_new_experts
            new_router = nn.Linear(old_in, new_out).to(device=device, dtype=model_dtype)
            if old_out > 0:
                new_router.weight.data[:old_out, :].copy_(old_classifier.weight.data)
                if old_classifier.bias is not None:
                    new_router.bias.data[:old_out].copy_(old_classifier.bias.data)
            mlp.router.classifier = new_router
            
            layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)

        # Log completion
        sample_mlp = model.model.layers[0].mlp
        total_experts = len(sample_mlp.scientific_experts) if hasattr(sample_mlp, 'scientific_experts') else 0
        router_dim = sample_mlp.router.classifier.out_features if hasattr(sample_mlp, 'router') else 0
        print_rank_0(f"[Upcycle-Refactored] Incremental conversion complete: {total_experts} experts, router_dim={router_dim}", local_rank)
        
        return model

    # non-incremental: build total experts = num_tasks * num_experts
    num_total_experts = int(num_tasks) * num_experts
    print_rank_0(f"[Upcycle-Refactored] ⚙️ Converting to Upcycling model with {num_total_experts} total experts for {num_tasks} tasks.", local_rank)

    for li, layer in enumerate(model.model.layers):
        if li % 2 != 0:
            pass
        print_rank_0(f"[Upcycle-Refactored] Converting layer {li} to MoE layer.", local_rank)

        mlp = layer.mlp
        mlp.original_forward = mlp.forward
        mlp.scientific_experts = nn.ModuleList([])

        if not hasattr(model.model.config, 'num_activated_experts'):
            model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))

        mlp.router = Router(model.model.config, num_total_experts).to(device=device, dtype=model_dtype)

        h = mlp.gate_proj.in_features
        H = mlp.gate_proj.out_features
        W_g = mlp.gate_proj.weight.data
        W_u = mlp.up_proj.weight.data
        W_d = mlp.down_proj.weight.data
        new_intermediate_size = H // num_experts

        for ei in range(num_total_experts):
            new_expert = Expert(model.model.config, new_intermediate_size).to(device=device, dtype=model_dtype)

            expert_idx_in_task = ei % num_experts
            start_col = expert_idx_in_task * new_intermediate_size
            end_col = (expert_idx_in_task + 1) * new_intermediate_size

            new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
            new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
            new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

            mlp.scientific_experts.append(new_expert)

        layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)

    # Log completion
    sample_mlp = model.model.layers[0].mlp
    router_dim = sample_mlp.router.classifier.out_features if hasattr(sample_mlp, 'router') else 0
    print_rank_0(f"[Upcycle-Refactored] Conversion complete: {num_total_experts} experts, router_dim={router_dim}", local_rank)

    return model


def convert_upcycle_model_variable(model, args, expert_count_by_layer: dict):
    """
    Convert the model to an Upcycling MoE model with VARIABLE expert counts per layer.
    
    This is needed for V4/V5 selective expansion, where different layers have different
    numbers of experts based on NAS decisions.
    
    Args:
        model: The base model to convert
        args: Arguments containing num_activated_experts, local_rank
        expert_count_by_layer: Dict mapping layer_idx -> num_experts for that layer
                              e.g., {6: 15, 9: 12, 12: 14, 13: 17, 15: 15, ...}
    
    Returns:
        model: The converted MoE model with variable expert counts per layer
    """
    local_rank = getattr(args, 'local_rank', 0)
    device = torch.device("cuda")
    model_dtype = next(model.parameters()).dtype
    base_num_experts = int(getattr(args, 'num_experts_per_task', 8))
    
    # Log conversion info
    print_rank_0(f"[Upcycle-Variable] Converting model with variable expert counts per layer", local_rank)
    print_rank_0(f"[Upcycle-Variable] Expert counts: {expert_count_by_layer}", local_rank)
    
    for li, layer in enumerate(model.model.layers):
        # Get expert count for this layer (default to base if not specified)
        num_experts_this_layer = expert_count_by_layer.get(li, base_num_experts)
        
        print_rank_0(f"[Upcycle-Variable] Converting layer {li}: {num_experts_this_layer} experts", local_rank)
        
        mlp = layer.mlp
        mlp.original_forward = mlp.forward
        mlp.scientific_experts = nn.ModuleList([])
        
        if not hasattr(model.model.config, 'num_activated_experts'):
            model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))
        
        # Create router with correct output dimension for this layer
        mlp.router = Router(model.model.config, num_experts_this_layer).to(device=device, dtype=model_dtype)
        
        h = mlp.gate_proj.in_features
        H = mlp.gate_proj.out_features
        W_g = mlp.gate_proj.weight.data
        W_u = mlp.up_proj.weight.data
        W_d = mlp.down_proj.weight.data
        
        # Calculate expert intermediate size based on base_num_experts (original FFN split)
        new_intermediate_size = H // base_num_experts
        
        # Create experts - more experts means we cycle through the FFN weights
        for ei in range(num_experts_this_layer):
            new_expert = Expert(model.model.config, new_intermediate_size).to(device=device, dtype=model_dtype)
            
            # Cycle through FFN segments
            expert_idx_in_task = ei % base_num_experts
            start_col = expert_idx_in_task * new_intermediate_size
            end_col = (expert_idx_in_task + 1) * new_intermediate_size
            
            new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
            new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
            new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
            
            mlp.scientific_experts.append(new_expert)
        
        layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
    
    # Log completion summary
    total_experts = sum(expert_count_by_layer.values())
    num_moe_layers = len(expert_count_by_layer)
    print_rank_0(f"[Upcycle-Variable] Conversion complete: {total_experts} total experts across {num_moe_layers} layers", local_rank)
    
    return model


# =============================================================================
# Upcycle Model Class
# =============================================================================

class Upcycle(CL_Base_Model):
    """
    Upcycling MoE Continual Learning Method.
    
    Supports two initialization modes for Task 0:
    - Non-expansion mode (task0_expansion_mode=False): 
      Dense -> 0~7 experts, all trainable
    - Expansion mode (task0_expansion_mode=True):
      Dense -> 0~7 (frozen) -> expand to 8~15 (trainable)
    
    Evaluation metrics (when enable_metrics=True):
    - CKA: Compare expanded experts vs original
    - Flatness: Evaluate loss landscape characteristics
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        
        self.current_task_id = 0
        self.task2expert_range = {}
        
        # MoE configuration
        self.num_experts_per_task = getattr(args, 'num_experts_per_task', 8)
        self.num_activated_experts = getattr(args, 'num_activated_experts', 2)
        
        # Upcycle control
        self.upcycle_interval = getattr(args, 'upcycle_interval', 4)
        raw_names = getattr(args, 'upcycle_task_names', [])
        if isinstance(raw_names, str):
            names = raw_names.split(',') if raw_names else []
        else:
            names = raw_names
        self.upcycle_task_names = set([n.strip() for n in names if n and n.strip()])
        
        # Router initialization method: 'random', 'average', 'zero_bias', 'scaled_random'
        self.router_init_method = getattr(args, 'router_init_method', 'random')

        # ==========================================================================
        # Task 0 Initialization Mode Configuration
        # ==========================================================================
        # If True: Task 0 creates experts 0~7 (frozen), then expands to 8~15 (trainable)
        # If False: Task 0 creates experts 0~7 (all trainable)
        self.task0_expansion_mode = getattr(args, 'task0_expansion_mode', False)
        
        # ==========================================================================
        # Evaluation Configuration
        # ==========================================================================
        self.enable_metrics = getattr(args, 'enable_expert_metrics', False)
        self.metric_num_batches = getattr(args, 'metric_num_batches', 5)
        self.metric_layer_idx = getattr(args, 'metric_layer_idx', 0)
        self.metric_all_layers = getattr(args, 'metric_all_layers', False)
        
        # Flatness configuration
        # Available methods (fastest to slowest):
        # - 'gradient_norm': ||∇L||₂, fastest approximation (~0.1s/expert)
        # - 'landscape': Random direction perturbation (~0.5s/expert)
        # - 'fisher_trace': Tr(F) ≈ Tr(H) (~1s/expert)
        # - 'hessian': Power iteration for λ_max (~5-10s/expert, most accurate)
        # - 'all': Run all methods for comparison
        self.flatness_method = getattr(args, 'flatness_method', 'gradient_norm')  # Default to fastest
        # Parse flatness_methods: can be string "method1,method2" or list
        raw_methods = getattr(args, 'flatness_methods', 'gradient_norm,landscape')
        if isinstance(raw_methods, str):
            if raw_methods.lower() == 'all':
                self.flatness_methods = ['gradient_norm', 'landscape', 'fisher_trace', 'hessian']
            else:
                self.flatness_methods = [m.strip() for m in raw_methods.split(',') if m.strip()]
        else:
            self.flatness_methods = raw_methods if raw_methods else ['gradient_norm', 'landscape']
        self.flatness_loss_type = getattr(args, 'flatness_loss_type', 'reconstruction')  # 'reconstruction' is faster
        self.sharpness_epsilon = getattr(args, 'sharpness_epsilon', 0.001)
        self.power_iterations = getattr(args, 'power_iterations', 10)  # Reduced for speed
        self.hutchinson_samples = getattr(args, 'hutchinson_samples', 5)
        self.flatness_batch_size = getattr(args, 'flatness_batch_size', 32)
        self.flatness_max_samples = getattr(args, 'flatness_max_samples', 128)
        self.flatness_training_batches = getattr(args, 'flatness_training_batches', 5)
        
        # Landscape method parameters (for fast evaluation)
        self.landscape_steps = getattr(args, 'landscape_steps', 5)
        self.landscape_multiplier = getattr(args, 'landscape_multiplier', 0.1)
        self.landscape_num_directions = getattr(args, 'landscape_num_directions', 2)
        
        # Enable visualization
        self.enable_flatness_visualization = getattr(args, 'enable_flatness_visualization', True)
        
        # Storage
        self.metric_results = {}
        self.original_dense_ffn = None  # Saved for evaluation
        self.expert_optimizer = None    # Separate optimizer for new experts
        
        # Cache for completed tasks' data (for CKA evaluation on old tasks)
        self.completed_task_cached_data = {}  # {task_id: cached_batches}
        
        # Device setup
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda", self.args.local_rank)
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Logging
        print_rank_0(f"[MoE-Upcycle] Initialized with {self.num_experts_per_task} experts/task, "
                     f"{self.num_activated_experts} activated", self.args.global_rank)
        if self.task0_expansion_mode:
            print_rank_0(f"[MoE-Upcycle] Task 0 Expansion Mode: 0~7 frozen, 8~15 trainable", self.args.global_rank)
        else:
            print_rank_0(f"[MoE-Upcycle] Task 0 Non-Expansion Mode: 0~7 trainable", self.args.global_rank)
        if self.enable_metrics:
            # Estimate time per expert-layer for each method
            time_estimates = {
                'gradient_norm': '~0.1s',
                'landscape': '~0.5s',
                'fisher_trace': '~1s',
                'hessian': '~5-10s'
            }
            method_times = [time_estimates.get(m, '?') for m in self.flatness_methods]
            print_rank_0(f"[MoE-Upcycle] Metrics ENABLED: methods={self.flatness_methods}", self.args.global_rank)
            print_rank_0(f"[MoE-Upcycle] Estimated time per expert-layer: {dict(zip(self.flatness_methods, method_times))}", self.args.global_rank)
            if FAST_FLATNESS_AVAILABLE:
                print_rank_0(f"[MoE-Upcycle] Fast flatness module: AVAILABLE", self.args.global_rank)
            else:
                print_rank_0(f"[MoE-Upcycle] Fast flatness module: NOT AVAILABLE (using legacy)", self.args.global_rank)

    # =========================================================================
    # Training Methods
    # =========================================================================
    
    def train_continual(self):
        """Main continual learning loop with upcycling."""
        # Support resuming from a specific task
        start_task = getattr(self.args, 'start_task', 0)
        if start_task > 0:
            print_rank_0(f"[MoE-Upcycle] Resuming from task {start_task}, skipping tasks 0-{start_task-1}", 
                        self.args.global_rank)
            # When resuming, we need to rebuild MoE structure for all previous tasks
            # then load weights from checkpoint
            self._restore_moe_structure_for_resume(start_task)
        
        for i_task, task in enumerate(self.train_task_list):
            # Skip completed tasks when resuming
            if i_task < start_task:
                print_rank_0(f"[MoE-Upcycle] Skipping task-{i_task}: {task} (already completed)", 
                            self.args.global_rank)
                continue
                
            self.current_task_id = i_task
            print_rank_0(f"[MoE-Upcycle] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

            # Decide whether to upcycle for this task
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True

            if do_upcycle:
                self.upcycle_one_task(task, i_task)
            else:
                print_rank_0(f"[MoE-Upcycle] Skipping upcycle for task-{i_task}", self.args.global_rank)

            # Train the model on the current task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Save the model state after training
            self.save_model(i_task)

    def _restore_moe_structure_for_resume(self, start_task: int):
        """
        Restore MoE structure when resuming from a checkpoint.
        
        When resuming from task N, the checkpoint contains MoE weights but 
        transformers loads only the base Llama structure. We need to:
        1. Rebuild MoE structure (experts + routers) for all tasks 0..N-1
        2. Load the MoE weights from the checkpoint
        """
        # Check if MoE structure already exists
        first_mlp = self.model.model.layers[0].mlp
        if hasattr(first_mlp, 'scientific_experts') and len(first_mlp.scientific_experts) > 0:
            print_rank_0(f"[Resume] MoE structure already exists with {len(first_mlp.scientific_experts)} experts", 
                        self.args.global_rank)
            return
        
        print_rank_0(f"[Resume] Rebuilding MoE structure for tasks 0-{start_task-1}...", self.args.global_rank)
        
        # Calculate total experts needed: start_task * num_experts_per_task
        total_experts_needed = start_task * self.num_experts_per_task
        
        # Step 1: Create Task 0 experts (from dense FFN)
        self._save_original_dense_ffn()
        if not self.task0_expansion_mode:
            self._upcycle_task0_non_expansion_mode()
        else:
            self._upcycle_task0_expansion_mode()
        
        # Step 2: Add experts for tasks 1..start_task-1
        for prev_task in range(1, start_task):
            print_rank_0(f"[Resume] Adding expert structure for task {prev_task}...", self.args.global_rank)
            self._upcycle_add_experts(prev_task)
        
        # Step 3: Load MoE weights from checkpoint
        ckpt_path = self.args.model_name_or_path
        ckpt_file = os.path.join(ckpt_path, "pytorch_model.bin")
        if os.path.exists(ckpt_file):
            print_rank_0(f"[Resume] Loading MoE weights from {ckpt_path}...", self.args.global_rank)
            state_dict = torch.load(ckpt_file, map_location='cpu')
            
            # Get the base model (handle DeepSpeed wrapping)
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Load MoE-related weights directly to model layers
            moe_keys_loaded = 0
            moe_keys_failed = 0
            
            for key, value in state_dict.items():
                if 'scientific_experts' not in key and 'router' not in key:
                    continue
                    
                try:
                    # Parse key: model.layers.X.mlp.scientific_experts.Y.weight_name
                    # or: model.layers.X.mlp.router.classifier.weight/bias
                    parts = key.split('.')
                    
                    # Find layer index
                    layer_idx = None
                    for i, p in enumerate(parts):
                        if p == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                            layer_idx = int(parts[i + 1])
                            break
                    
                    if layer_idx is None:
                        continue
                    
                    mlp = base_model.model.layers[layer_idx].mlp
                    
                    if 'scientific_experts' in key:
                        # Parse expert index
                        expert_idx = None
                        for i, p in enumerate(parts):
                            if p == 'scientific_experts' and i + 1 < len(parts) and parts[i + 1].isdigit():
                                expert_idx = int(parts[i + 1])
                                break
                        
                        if expert_idx is None or expert_idx >= len(mlp.scientific_experts):
                            continue
                        
                        expert = mlp.scientific_experts[expert_idx]
                        # Get the projection layer (gate_proj, up_proj, down_proj)
                        proj_name = parts[-2]  # e.g., 'gate_proj'
                        if hasattr(expert, proj_name):
                            proj = getattr(expert, proj_name)
                            if hasattr(proj, 'weight') and proj.weight.shape == value.shape:
                                proj.weight.data.copy_(value.to(proj.weight.device))
                                moe_keys_loaded += 1
                            else:
                                moe_keys_failed += 1
                    
                    elif 'router' in key:
                        # Router weights: model.layers.X.mlp.router.classifier.weight/bias
                        if hasattr(mlp, 'router') and hasattr(mlp.router, 'classifier'):
                            classifier = mlp.router.classifier
                            param_name = parts[-1]  # weight or bias
                            if hasattr(classifier, param_name):
                                param = getattr(classifier, param_name)
                                if param.shape == value.shape:
                                    param.data.copy_(value.to(param.device))
                                    moe_keys_loaded += 1
                                else:
                                    moe_keys_failed += 1
                                    
                except Exception as e:
                    moe_keys_failed += 1
            
            print_rank_0(f"[Resume] Loaded {moe_keys_loaded} MoE weights, {moe_keys_failed} failed", 
                        self.args.global_rank)
        else:
            print_rank_0(f"[Resume] WARNING: No pytorch_model.bin found at {ckpt_path}", self.args.global_rank)
        
        # Verify structure
        first_mlp = self.model.model.layers[0].mlp
        if hasattr(first_mlp, 'scientific_experts'):
            print_rank_0(f"[Resume] MoE structure restored: {len(first_mlp.scientific_experts)} experts per layer", 
                        self.args.global_rank)

    def upcycle_one_task(self, task, i_task):
        """
        Adds and initializes new experts for the current task.
        
        For Task 0:
        - Expansion mode: Create 0~7 (frozen), then expand to 8~15 (trainable)
        - Non-expansion mode: Create 0~7 (trainable)
        
        For Task N (N > 0):
        - Expand by adding new experts for the task
        """
        if i_task in self.task2expert_range:
            print_rank_0(f"Task {i_task} already has experts. Skipping.", self.args.global_rank)
            return

        print_rank_0(f"Upcycling for task: {task} (Task ID: {i_task})", self.args.global_rank)
        
        # Save original dense FFN for evaluation (before any conversion)
        if i_task == 0 and self.original_dense_ffn is None:
            self._save_original_dense_ffn()

        # Task 0 handling
        if i_task == 0:
            if self.task0_expansion_mode:
                self._upcycle_task0_expansion_mode()
            else:
                self._upcycle_task0_non_expansion_mode()
        else:
            # Task N (N > 0): Add new experts
            self._upcycle_add_experts(i_task)
        
        # Create optimizer for new expert parameters
        self._create_expert_optimizer(i_task)

    def _save_original_dense_ffn(self):
        """Save original dense FFN weights for evaluation (stored on CPU to save GPU memory)."""
        self.original_dense_ffn = {}
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            self.original_dense_ffn[i] = {
                'gate_proj': mlp.gate_proj.weight.data.clone().cpu(),
                'up_proj': mlp.up_proj.weight.data.clone().cpu(),
                'down_proj': mlp.down_proj.weight.data.clone().cpu()
            }
        print_rank_0(f"Saved original dense FFN for {len(self.original_dense_ffn)} layers (on CPU)", self.args.global_rank)

    def _upcycle_task0_non_expansion_mode(self):
        """
        Task 0 Non-Expansion Mode: Dense -> 0~7 experts (all trainable)
        """
        print_rank_0("[Task 0 Non-Expansion] Creating experts 0~7 from dense FFN (trainable)", self.args.global_rank)
        
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            mlp.original_forward = mlp.forward
            mlp.scientific_experts = nn.ModuleList([])
            self.model.model.config.num_activated_experts = self.num_activated_experts
            
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            new_intermediate_size = H // self.num_experts_per_task
            
            # Create experts 0~7
            for ei in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(device=self.device, dtype=self.model_dtype)
                start_col = ei * new_intermediate_size
                end_col = (ei + 1) * new_intermediate_size
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                mlp.scientific_experts.append(new_expert)
            
            # Initialize router
            mlp.router = Router(self.model.model.config, self.num_experts_per_task).to(device=self.device, dtype=self.model_dtype)
            self._init_router_new_experts(mlp.router.classifier, 0, self.num_experts_per_task, None)
            
            # Patch forward method
            layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
        
        # Record expert range for task 0
        self.task2expert_range[0] = range(0, self.num_experts_per_task)
        print_rank_0(f"[Task 0 Non-Expansion] Created {self.num_experts_per_task} trainable experts", self.args.global_rank)

    def _upcycle_task0_expansion_mode(self):
        """
        Task 0 Expansion Mode: Dense -> 0~7 (frozen) -> expand to 8~15 (trainable)
        """
        print_rank_0("[Task 0 Expansion] Step 1: Creating experts 0~7 from dense FFN (frozen)", self.args.global_rank)
        
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            mlp.original_forward = mlp.forward
            mlp.scientific_experts = nn.ModuleList([])
            self.model.model.config.num_activated_experts = self.num_activated_experts
            
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            new_intermediate_size = H // self.num_experts_per_task
            
            # Create experts 0~7 (will be frozen)
            for ei in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(device=self.device, dtype=self.model_dtype)
                start_col = ei * new_intermediate_size
                end_col = (ei + 1) * new_intermediate_size
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                mlp.scientific_experts.append(new_expert)
            
            # Initialize router for 0~7
            mlp.router = Router(self.model.model.config, self.num_experts_per_task).to(device=self.device, dtype=self.model_dtype)
            self._init_router_new_experts(mlp.router.classifier, 0, self.num_experts_per_task, None)
            
            # Patch forward method
            layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
        
        print_rank_0("[Task 0 Expansion] Step 2: Expanding to experts 8~15 (trainable)", self.args.global_rank)
        
        # Step 2: Expand to 8~15
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            new_intermediate_size = H // self.num_experts_per_task
            
            total_before = len(mlp.scientific_experts)
            
            # Create experts 8~15 (trainable)
            for ei in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(device=self.device, dtype=self.model_dtype)
                start_col = ei * new_intermediate_size
                end_col = (ei + 1) * new_intermediate_size
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                mlp.scientific_experts.append(new_expert)
            
            # Expand router
            total_after = len(mlp.scientific_experts)
            new_router_classifier = nn.Linear(h, total_after, device=self.device, dtype=self.model_dtype)
            new_router_classifier.weight.data[:total_before, :] = mlp.router.classifier.weight.data
            if mlp.router.classifier.bias is not None:
                new_router_classifier.bias.data[:total_before] = mlp.router.classifier.bias.data
            self._init_router_new_experts(new_router_classifier, total_before, total_after, mlp.router.classifier)
            mlp.router.classifier = new_router_classifier
        
        # Record: Task 0 trains experts 8~15, not 0~7
        self.task2expert_range[0] = range(self.num_experts_per_task, 2 * self.num_experts_per_task)
        print_rank_0(f"[Task 0 Expansion] Experts 0~7 frozen, 8~15 trainable", self.args.global_rank)

    def _upcycle_add_experts(self, i_task):
        """Add new experts for task N (N > 0)."""
        total_before = len(self.model.model.layers[0].mlp.scientific_experts)
        print_rank_0(f"Adding {self.num_experts_per_task} experts for task {i_task} (before: {total_before})", self.args.global_rank)
        
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            new_intermediate_size = H // self.num_experts_per_task
            
            old_count = len(mlp.scientific_experts)
            
            # Add new experts
            for ei in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(device=self.device, dtype=self.model_dtype)
                start_col = ei * new_intermediate_size
                end_col = (ei + 1) * new_intermediate_size
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                mlp.scientific_experts.append(new_expert)
            
            # Expand router
            new_count = len(mlp.scientific_experts)
            new_router_classifier = nn.Linear(h, new_count, device=self.device, dtype=self.model_dtype)
            new_router_classifier.weight.data[:old_count, :] = mlp.router.classifier.weight.data
            if mlp.router.classifier.bias is not None:
                new_router_classifier.bias.data[:old_count] = mlp.router.classifier.bias.data
            self._init_router_new_experts(new_router_classifier, old_count, new_count, mlp.router.classifier)
            mlp.router.classifier = new_router_classifier
        
        # Record expert range
        start_idx = total_before
        end_idx = start_idx + self.num_experts_per_task
        self.task2expert_range[i_task] = range(start_idx, end_idx)
        print_rank_0(f"Task {i_task} experts: {start_idx}~{end_idx-1}", self.args.global_rank)

    def _init_router_new_experts(self, new_classifier: nn.Linear, old_num: int, new_num: int, old_classifier: nn.Linear = None):
        """Initialize router weights for newly added experts."""
        num_new = new_num - old_num
        if num_new <= 0:
            return
        
        method = self.router_init_method
        
        if method == 'average' and old_classifier is not None and old_num > 0:
            avg_weight = old_classifier.weight.data.mean(dim=0, keepdim=True)
            new_classifier.weight.data[old_num:, :] = avg_weight.expand(num_new, -1)
            if new_classifier.bias is not None and old_classifier.bias is not None:
                new_classifier.bias.data[old_num:] = old_classifier.bias.data.mean()
        elif method == 'zero_bias' and new_classifier.bias is not None:
            new_classifier.bias.data[old_num:] = -2.0
        elif method == 'scaled_random':
            with torch.no_grad():
                new_classifier.weight.data[old_num:, :] *= 0.1
                if new_classifier.bias is not None:
                    new_classifier.bias.data[old_num:] *= 0.1
        # Default: 'random' - use PyTorch default initialization

    def _create_expert_optimizer(self, i_task):
        """Create separate optimizer for new expert parameters."""
        if i_task not in self.task2expert_range:
            return
        
        expert_range = self.task2expert_range[i_task]
        new_params = []
        
        for layer in self.model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, 'scientific_experts'):
                for expert_idx in expert_range:
                    if expert_idx < len(mlp.scientific_experts):
                        for param in mlp.scientific_experts[expert_idx].parameters():
                            new_params.append(param)
                if hasattr(mlp, 'router'):
                    for param in mlp.router.classifier.parameters():
                        new_params.append(param)
        
        if new_params:
            lr = getattr(self.args, 'learning_rate', 1e-5)
            self.expert_optimizer = torch.optim.AdamW(new_params, lr=float(lr), weight_decay=0.0, betas=(0.9, 0.95))
            print_rank_0(f"Created optimizer for {len(new_params)} new parameters", self.args.global_rank)

    def freeze_non_current_task_params(self, i_task):
        """Freeze all parameters except current task's experts and router."""
        # Freeze all experts and routers first
        for layer in self.model.model.layers:
            mlp = layer.mlp
            for expert in getattr(mlp, "scientific_experts", []):
                for param in expert.parameters():
                    param.requires_grad = False
            if hasattr(mlp, "router"):
                for param in mlp.router.parameters():
                    param.requires_grad = False
            # Freeze MLP backbone
            for name, param in mlp.named_parameters():
                if not name.startswith("scientific_experts") and not name.startswith("router"):
                    param.requires_grad = False

        # Unfreeze current task's experts and router
        expert_range = self.task2expert_range.get(i_task)
        if expert_range is None:
            return

        print_rank_0(f"Unfreezing experts {list(expert_range)} for task {i_task}", self.args.global_rank)
        
        for i, layer in enumerate(self.model.model.layers):
            if not hasattr(layer.mlp, 'scientific_experts'):
                continue
            for expert_idx in expert_range:
                if expert_idx < len(layer.mlp.scientific_experts):
                    for param in layer.mlp.scientific_experts[expert_idx].parameters():
                        param.requires_grad = True
            
            # Unfreeze router but mask gradients for non-current experts
            for param in layer.mlp.router.classifier.parameters():
                param.requires_grad = True
                grad_mask = torch.zeros_like(param.data)
                if grad_mask.dim() == 2:
                    grad_mask[expert_range.start:expert_range.stop, :] = 1
                elif grad_mask.dim() == 1:
                    grad_mask[expert_range.start:expert_range.stop] = 1
                param.register_hook(lambda grad, mask=grad_mask: grad * mask)

    def train_one_task(self, task, i_task, epochs):
        """Train on a single task."""
        dataloader_train = self.train_task_list[task]
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        # Evaluate metrics BEFORE training (if enabled)
        if self.enable_metrics:
            if i_task == 0:
                # Task 0: evaluate MoE vs Dense
                self._evaluate_before_training(task, dataloader_train)
            else:
                # Task N (N > 0): CKA on old tasks, Flatness on new task
                self._evaluate_task_n_before_training(task, i_task, dataloader_train)

        # Freeze non-current parameters
        self.freeze_non_current_task_params(i_task)

        global_step = 0
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader_train):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Task-{i_task} Epoch-{epoch} loss={loss.item():.4f}")
                
                self.model.backward(loss)
                self.model.step()
                
                if self.expert_optimizer is not None:
                    self.expert_optimizer.step()
                    self.expert_optimizer.zero_grad()
                
                global_step += 1
        
        # Cache current task data for future CKA evaluation (on old tasks)
        if self.enable_metrics:
            self._cache_task_data_for_cka(task, i_task, dataloader_train)

    # =========================================================================
    # Evaluation Methods
    # =========================================================================
    
    def _cache_task_data_for_cka(self, task, i_task, dataloader):
        """Cache task data for future CKA evaluation on old tasks."""
        print_rank_0(f"[Metrics] Caching Task {i_task} data for future CKA evaluation...", self.args.global_rank)
        cached_batches = self._cache_dataloader_batches(dataloader, self.metric_num_batches)
        self.completed_task_cached_data[i_task] = {
            'task_name': task,
            'cached_batches': cached_batches
        }
        print_rank_0(f"[Metrics] Cached {len(cached_batches)} batches for Task {i_task}", self.args.global_rank)
    
    def _get_old_tasks_data(self, current_task_id: int) -> list:
        """Get cached data from all old tasks (Task 0 ~ Task N-1)."""
        all_cached_batches = []
        for task_id in range(current_task_id):
            if task_id in self.completed_task_cached_data:
                all_cached_batches.extend(self.completed_task_cached_data[task_id]['cached_batches'])
        return all_cached_batches
    
    def _evaluate_before_training(self, task, dataloader):
        """Evaluate metrics before Task 0 training starts."""
        if self.args.global_rank != 0:
            return
        
        if self.task0_expansion_mode:
            print_rank_0("\n" + "="*60, 0)
            print_rank_0("[Pre-Training Baseline] Expansion Mode Initialization Metrics", 0)
            print_rank_0("="*60, 0)
            results = self._evaluate_task_minus_1_expansion_mode(task, dataloader)
        else:
            print_rank_0("\n" + "="*60, 0)
            print_rank_0("[Pre-Training Baseline] Non-Expansion Mode Initialization Metrics", 0)
            print_rank_0("="*60, 0)
            results = self._evaluate_task_minus_1_non_expansion_mode(task, dataloader)
        
        if results:
            self._save_metric_results(0, "initialization_baseline_metrics", results)
    
    def _evaluate_task_n_before_training(self, task, i_task, dataloader):
        """
        Evaluate metrics before Task N (N > 0) training starts.
        
        - CKA: Uses OLD task data to measure knowledge retention after expansion
        - Flatness: Uses NEW task data to measure trainability of old experts
        """
        if self.args.global_rank != 0:
            return
        
        eval_start_time = time.time()
        
        print_rank_0(f"\n" + "="*60, 0)
        print_rank_0(f"[Task {i_task}] Pre-Training Evaluation", 0)
        print_rank_0("="*60, 0)
        print_rank_0(f"   CKA: Using data from old tasks (0~{i_task-1}) to measure knowledge retention", 0)
        print_rank_0(f"   Flatness: Using Task {i_task} data to measure old experts trainability", 0)
        
        results = {
            'task_id': i_task,
            'task_name': task,
            'cka_on_old_tasks': {},
            'old_experts_flatness_on_new_task': {},
            'timing': {}
        }
        
        layer_indices = self._get_layers_to_evaluate()
        
        # =====================================================================
        # 1. CKA: Evaluate on OLD task data (knowledge retention)
        # =====================================================================
        old_tasks_data = self._get_old_tasks_data(i_task)
        cka_time = 0
        if old_tasks_data:
            print_rank_0(f"\n[CKA] Knowledge Retention: Using {len(old_tasks_data)} batches from old tasks", 0)
            cka_start = time.time()
            cka_results = self._evaluate_cka_on_old_tasks(layer_indices, old_tasks_data, i_task)
            cka_time = time.time() - cka_start
            results['cka_on_old_tasks'] = cka_results
            results['timing']['cka_seconds'] = cka_time
            print_rank_0(f"   CKA completed in {cka_time:.1f}s", 0)
        else:
            print_rank_0(f"\n[CKA] Skipped: No old task data cached", 0)
        
        # =====================================================================
        # 2. Flatness: Evaluate old experts on NEW task data (trainability)
        # =====================================================================
        print_rank_0(f"\n[Flatness] Old Experts on Task {i_task} Data", 0)
        flatness_start = time.time()
        new_task_cached = self._cache_dataloader_batches(dataloader, self.metric_num_batches)
        
        # Get old expert indices (all experts from tasks 0 ~ i_task-1)
        old_expert_indices = []
        for prev_task_id in range(i_task):
            if prev_task_id in self.task2expert_range:
                old_expert_indices.extend(list(self.task2expert_range[prev_task_id]))
        
        flatness_time = 0
        if old_expert_indices:
            flatness_results = self._evaluate_experts_flatness(
                layer_indices, new_task_cached, range(min(old_expert_indices), max(old_expert_indices) + 1),
                scenario_name=f'old_experts_on_task{i_task}',
                temporary_train=False, dataloader=dataloader  # Skip temp train due to DeepSpeed ZeRO
            )
            flatness_time = time.time() - flatness_start
            results['old_experts_flatness_on_new_task'] = flatness_results
            results['timing']['flatness_seconds'] = flatness_time
            print_rank_0(f"   Flatness completed in {flatness_time:.1f}s", 0)
        
        total_time = time.time() - eval_start_time
        results['timing']['total_seconds'] = total_time
        print_rank_0(f"\n[Timing] Task {i_task} evaluation: {total_time:.1f}s (CKA: {cka_time:.1f}s, Flatness: {flatness_time:.1f}s)", 0)
        
        if results:
            self._save_metric_results(i_task, "before_training_metrics", results)
        
        self._clear_cuda_cache()
    
    def _evaluate_cka_on_old_tasks(self, layer_indices: list, old_tasks_cached_batches: list, current_task_id: int) -> dict:
        """
        Evaluate CKA between current MoE (after expansion) and previous MoE (before expansion)
        using OLD task data. This measures knowledge retention.
        
        Comparison:
        - Old experts only (before expansion) output on old task data
        - All experts (after expansion) output on old task data
        """
        results = {}
        
        from evaluations.representation_metrics import RepresentationAlignmentEvaluator
        rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
        
        # Get old expert indices (experts from previous tasks)
        old_expert_indices = []
        for prev_task_id in range(current_task_id):
            if prev_task_id in self.task2expert_range:
                old_expert_indices.extend(list(self.task2expert_range[prev_task_id]))
        
        # Get new expert indices (current task's experts)
        new_expert_indices = list(self.task2expert_range.get(current_task_id, range(0)))
        
        cka_pbar = tqdm(layer_indices, desc="[CKA] Layers", leave=False, disable=(self.args.global_rank != 0))
        
        for layer_idx in cka_pbar:
            cka_pbar.set_postfix({'layer': layer_idx})
            mlp = self._get_mlp(layer_idx)
            hidden_states = self._prepare_hidden_states(old_tasks_cached_batches, layer_idx)
            
            if hidden_states is None:
                continue
            
            with torch.no_grad():
                # Compute output using OLD experts only (simulating pre-expansion state)
                old_experts_output = self._compute_moe_output_with_experts(
                    mlp, hidden_states, old_expert_indices
                )
                
                # Compute output using ALL experts (current state after expansion)
                all_experts_output = self._compute_moe_output_with_experts(
                    mlp, hidden_states, old_expert_indices + new_expert_indices
                )
                
                # Compute CKA
                old_flat = old_experts_output.view(-1, old_experts_output.size(-1))
                all_flat = all_experts_output.view(-1, all_experts_output.size(-1))
                
                cka_score = rep_evaluator._compute_linear_cka(old_flat, all_flat)
            
            results[f'layer_{layer_idx}'] = {
                'cka_score': float(cka_score),
                'old_experts': old_expert_indices,
                'new_experts': new_expert_indices,
                'data_source': f'old_tasks_0_to_{current_task_id-1}',
                'interpretation': 'high=knowledge_retained, low=knowledge_degraded'
            }
            
            del old_experts_output, all_experts_output
            self._clear_cuda_cache()
        
        cka_pbar.close()
        return results
    
    def _compute_moe_output_with_experts(self, mlp, hidden_states: torch.Tensor, expert_indices: list) -> torch.Tensor:
        """
        Compute MoE output using only specified experts.
        This simulates what the output would be with a subset of experts.
        """
        # Shared expert output (original FFN)
        shared_output = mlp.original_forward(hidden_states)
        
        if not expert_indices:
            return shared_output
        
        # Compute expert output
        input_was_2d = hidden_states.dim() == 2
        if input_was_2d:
            num_tokens, hidden_dim = hidden_states.shape
            x_3d = hidden_states.unsqueeze(0)
        else:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            num_tokens = batch_size * seq_len
            x_3d = hidden_states
        
        # Get routing decisions (using current router)
        routing_weights, selected_experts = mlp.router(x_3d)
        flat_x = x_3d.view(num_tokens, hidden_dim)
        expert_output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for expert_idx in expert_indices:
            if expert_idx >= len(mlp.scientific_experts):
                continue
            
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue
            
            expert_input = flat_x[token_indices]
            expert_out = mlp.scientific_experts[expert_idx](expert_input)
            token_weights = (routing_weights[token_indices] * expert_mask[token_indices].float()).sum(dim=-1, keepdim=True)
            expert_output[token_indices] += token_weights * expert_out
        
        if input_was_2d:
            expert_output = expert_output.view(num_tokens, hidden_dim)
            if shared_output.dim() == 3:
                shared_output = shared_output.squeeze(0)
        else:
            expert_output = expert_output.view(batch_size, seq_len, hidden_dim)
        
        return shared_output + expert_output

    def _evaluate_task_minus_1_expansion_mode(self, task, dataloader) -> dict:
        """
        Initialization Baseline Evaluation in Expansion Mode:
        - CKA: Compare experts 8~15 vs 0~7 (expected ~1.0 since both from same FFN)
        - Flatness: Evaluate experts 0~7 if trained on task 0 data
        """
        eval_start_time = time.time()
        
        results = {
            'eval_type': 'initialization_baseline',
            'task_name': task,
            'expansion_mode': True,
            'cka_new_vs_old_experts': {},
            'old_experts_flatness': {},
            'timing': {}
        }
        
        layer_indices = self._get_layers_to_evaluate()
        cached_batches = self._cache_dataloader_batches(dataloader, self.metric_num_batches)
        
        # 1. CKA: Compare experts 8~15 vs 0~7
        # Note: At initialization, both expert groups are derived from the same dense FFN,
        # so they should produce nearly identical outputs (CKA ~1.0 or undefined/-1.0)
        print_rank_0("\n[CKA] Initialization Similarity: New Experts (8~15) vs Base Experts (0~7)", 0)
        print_rank_0("      Note: Both derived from same FFN, high similarity expected", 0)
        
        from evaluations.representation_metrics import RepresentationAlignmentEvaluator
        rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
        
        cka_start = time.time()
        cka_pbar = tqdm(layer_indices, desc="[CKA] Layers", leave=False, disable=(self.args.global_rank != 0))
        
        for layer_idx in cka_pbar:
            cka_pbar.set_postfix({'layer': layer_idx})
            mlp = self._get_mlp(layer_idx)
            hidden_states = self._prepare_hidden_states(cached_batches, layer_idx)
            if hidden_states is None:
                continue
            
            n = self.num_experts_per_task
            if len(mlp.scientific_experts) < 2 * n:
                continue
            
            # Compute outputs for old experts (0~7) and new experts (8~15)
            with torch.no_grad():
                old_outputs = [mlp.scientific_experts[i](hidden_states) for i in range(n)]
                new_outputs = [mlp.scientific_experts[i](hidden_states) for i in range(n, 2*n)]
                
                old_avg = torch.stack(old_outputs, dim=0).mean(dim=0).view(-1, hidden_states.size(-1))
                new_avg = torch.stack(new_outputs, dim=0).mean(dim=0).view(-1, hidden_states.size(-1))
                
                cka_score = rep_evaluator._compute_linear_cka(old_avg, new_avg)
            
            # Interpret the result
            if cka_score < 0:
                interpretation = "identical_outputs (variance~0, expected at init)"
            else:
                interpretation = f"similarity={cka_score:.4f}"
            
            results['cka_new_vs_old_experts'][f'layer_{layer_idx}'] = {
                'cka_score': float(cka_score),
                'old_experts': list(range(n)),
                'new_experts': list(range(n, 2*n)),
                'interpretation': interpretation
            }
        
        cka_pbar.close()
        cka_time = time.time() - cka_start
        results['timing']['cka_seconds'] = cka_time
        print_rank_0(f"   CKA completed in {cka_time:.1f}s", 0)
        
        # 2. Flatness: Evaluate old experts (0~7) on task 0 data
        # NOTE: We evaluate flatness WITHOUT temporary training because:
        # 1. DeepSpeed ZeRO Stage 2 doesn't support manual backward on dynamic params
        # 2. Evaluating untrained state reflects experts' adaptation potential to new data
        print_rank_0("\n[Flatness] Base Experts (0~7) - Current State on Task 0 Data", 0)
        flatness_start = time.time()
        flatness_results = self._evaluate_experts_flatness(
            layer_indices, cached_batches, range(self.num_experts_per_task), 
            scenario_name='base_experts_task0', temporary_train=False, dataloader=dataloader
        )
        flatness_time = time.time() - flatness_start
        results['old_experts_flatness'] = flatness_results
        results['timing']['flatness_seconds'] = flatness_time
        print_rank_0(f"   Flatness completed in {flatness_time:.1f}s", 0)
        
        total_time = time.time() - eval_start_time
        results['timing']['total_seconds'] = total_time
        print_rank_0(f"\n[Timing] Total evaluation: {total_time:.1f}s (CKA: {cka_time:.1f}s, Flatness: {flatness_time:.1f}s)", 0)
        
        self._clear_cuda_cache()
        return results

    def _evaluate_task_minus_1_non_expansion_mode(self, task, dataloader) -> dict:
        """
        Initialization Baseline Evaluation in Non-Expansion Mode:
        - CKA: Compare MoE experts (0~7) vs original Dense FFN
        - Flatness: Evaluate Dense FFN if trained on task 0 data
        """
        eval_start_time = time.time()
        
        if self.original_dense_ffn is None:
            print_rank_0("[Metrics] Original dense FFN not saved, skipping", 0)
            return {}
        
        results = {
            'eval_type': 'initialization_baseline',
            'task_name': task,
            'expansion_mode': False,
            'cka_moe_vs_dense': {},
            'dense_ffn_flatness': {},
            'timing': {}
        }
        
        layer_indices = self._get_layers_to_evaluate()
        cached_batches = self._cache_dataloader_batches(dataloader, self.metric_num_batches)
        
        from evaluations.representation_metrics import RepresentationAlignmentEvaluator
        rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
        
        # 1. CKA: Compare MoE experts vs Dense FFN
        print_rank_0("\n[CKA] MoE Experts (0~7) vs Original Dense FFN", 0)
        print_rank_0("      Note: Experts are split from FFN, high similarity expected", 0)
        
        cka_start = time.time()
        valid_layers = [l for l in layer_indices if l in self.original_dense_ffn]
        cka_pbar = tqdm(valid_layers, desc="[CKA] Layers", leave=False, disable=(self.args.global_rank != 0))
        
        for layer_idx in cka_pbar:
            cka_pbar.set_postfix({'layer': layer_idx})
            mlp = self._get_mlp(layer_idx)
            hidden_states = self._prepare_hidden_states(cached_batches, layer_idx)
            if hidden_states is None:
                continue
            
            with torch.no_grad():
                # MoE experts output (average of 0~7)
                moe_outputs = [mlp.scientific_experts[i](hidden_states) for i in range(min(self.num_experts_per_task, len(mlp.scientific_experts)))]
                moe_avg = torch.stack(moe_outputs, dim=0).mean(dim=0).view(-1, hidden_states.size(-1))
                
                # Dense FFN output
                dense_weights = self.original_dense_ffn[layer_idx]
                dense_output = self._compute_dense_output(hidden_states, dense_weights, mlp)
                dense_flat = dense_output.view(-1, dense_output.size(-1))
                
                cka_score = rep_evaluator._compute_linear_cka(moe_avg, dense_flat)
            
            # Interpret the result
            if cka_score < 0:
                interpretation = "identical_outputs (variance~0, expected at init)"
            else:
                interpretation = f"similarity={cka_score:.4f}"
            
            results['cka_moe_vs_dense'][f'layer_{layer_idx}'] = {
                'cka_score': float(cka_score),
                'interpretation': interpretation
            }
        
        cka_pbar.close()
        cka_time = time.time() - cka_start
        results['timing']['cka_seconds'] = cka_time
        print_rank_0(f"   CKA completed in {cka_time:.1f}s", 0)
        
        # 2. Flatness: Evaluate Dense FFN
        print_rank_0("\n[Flatness] Original Dense FFN on Task 0 Data", 0)
        flatness_start = time.time()
        flatness_pbar = tqdm(valid_layers, desc="[Flatness] Layers", leave=False, disable=(self.args.global_rank != 0))
        
        for layer_idx in flatness_pbar:
            flatness_pbar.set_postfix({'layer': layer_idx})
            hidden_states = self._prepare_hidden_states(cached_batches, layer_idx)
            if hidden_states is None:
                continue
            
            dense_flatness = self._evaluate_dense_ffn_flatness(layer_idx, hidden_states, cached_batches)
            results['dense_ffn_flatness'][f'layer_{layer_idx}'] = dense_flatness
        
        flatness_pbar.close()
        flatness_time = time.time() - flatness_start
        results['timing']['flatness_seconds'] = flatness_time
        print_rank_0(f"   Flatness completed in {flatness_time:.1f}s", 0)
        
        total_time = time.time() - eval_start_time
        results['timing']['total_seconds'] = total_time
        print_rank_0(f"\n[Timing] Total evaluation: {total_time:.1f}s (CKA: {cka_time:.1f}s, Flatness: {flatness_time:.1f}s)", 0)
        
        self._clear_cuda_cache()
        return results

    def _evaluate_experts_flatness(
        self, 
        layer_indices: list, 
        cached_batches: list, 
        expert_indices: range,
        scenario_name: str,
        temporary_train: bool = False,
        dataloader = None
    ) -> dict:
        """
        Evaluate flatness for specified experts using fast methods.
        
        Uses the fast_flatness module for efficient evaluation across all layers and experts.
        """
        # Use fast evaluation if available
        if FAST_FLATNESS_AVAILABLE:
            return self._evaluate_experts_flatness_fast(
                layer_indices, cached_batches, expert_indices, scenario_name
            )
        
        # Fallback to original (slower) implementation
        return self._evaluate_experts_flatness_legacy(
            layer_indices, cached_batches, expert_indices, scenario_name, 
            temporary_train, dataloader
        )
    
    def _evaluate_experts_flatness_fast(
        self,
        layer_indices: list,
        cached_batches: list,
        expert_indices: range,
        scenario_name: str
    ) -> dict:
        """
        Fast flatness evaluation using multiple methods in parallel.
        
        Evaluates all layers and experts efficiently using the fast_flatness module.
        """
        print_rank_0(f"\n   [Fast Flatness] {scenario_name}", 0)
        print_rank_0(f"   Methods: {self.flatness_methods}", 0)
        print_rank_0(f"   Layers: {len(layer_indices)}, Experts: {len(list(expert_indices))}", 0)
        
        start_time = time.time()
        
        # Use fast evaluation
        results = evaluate_all_experts_fast(
            model=self.model,
            layer_indices=layer_indices,
            expert_range=expert_indices,
            cached_batches=cached_batches,
            device=self.device,
            methods=self.flatness_methods,
            landscape_steps=self.landscape_steps,
            landscape_multiplier=self.landscape_multiplier,
            max_samples=self.flatness_max_samples,
            verbose=True,
            rank=self.args.global_rank
        )
        
        total_time = time.time() - start_time
        print_rank_0(f"   [Fast Flatness] Completed in {total_time:.1f}s", 0)
        
        # Generate visualization if enabled
        if self.enable_flatness_visualization and self.args.global_rank == 0:
            self._generate_flatness_visualization(results, scenario_name)
        
        return results
    
    def _generate_flatness_visualization(self, results: dict, scenario_name: str):
        """Generate and save flatness visualization."""
        if not FAST_FLATNESS_AVAILABLE:
            return
        
        try:
            output_dir = getattr(self.args, 'output_dir', None)
            if output_dir is None:
                return
            
            viz_dir = os.path.join(output_dir, 'flatness_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            visualizer = FlatnessVisualizer(output_dir=viz_dir)
            
            # Create heatmap for each method
            for method in self.flatness_methods:
                metric_key = {
                    'gradient_norm': 'gradient_norm',
                    'landscape': 'avg_loss_change',
                    'fisher_trace': 'fisher_trace_normalized',
                    'hessian': 'lambda_max'
                }.get(method, 'gradient_norm')
                
                safe_name = scenario_name.replace(' ', '_').replace('~', '-')
                save_path = os.path.join(viz_dir, f'{safe_name}_{method}_heatmap.png')
                
                visualizer.create_heatmap(
                    results=results,
                    metric_key=metric_key,
                    title=f'{scenario_name} - {method}',
                    save_path=save_path
                )
            
            print_rank_0(f"   [Visualization] Saved to {viz_dir}", 0)
            
        except Exception as e:
            print_rank_0(f"   [Warning] Visualization failed: {e}", 0)
    
    def _evaluate_experts_flatness_legacy(
        self, 
        layer_indices: list, 
        cached_batches: list, 
        expert_indices: range,
        scenario_name: str,
        temporary_train: bool = False,
        dataloader = None
    ) -> dict:
        """Legacy flatness evaluation (slower, uses Hessian)."""
        results = {}
        num_experts = len(list(expert_indices))
        total_evaluations = len(layer_indices) * num_experts
        
        # If temporary training requested, save and restore model state
        if temporary_train:
            saved_state = self._save_model_state()
            print_rank_0(f"   Temporary training {num_experts} experts for {self.flatness_training_batches} batches...", 0)
            self._temporary_train_experts(expert_indices, dataloader, num_batches=self.flatness_training_batches)
        
        try:
            # Create overall progress bar
            pbar = tqdm(
                total=total_evaluations, 
                desc=f"[Flatness] {scenario_name}", 
                leave=False, 
                disable=(self.args.global_rank != 0)
            )
            
            for layer_idx in layer_indices:
                mlp = self._get_mlp(layer_idx)
                hidden_states = self._prepare_hidden_states(cached_batches, layer_idx)
                if hidden_states is None:
                    pbar.update(num_experts)
                    continue
                
                # Get routed tokens for each expert
                routed_dict = self._get_all_experts_routed_tokens(hidden_states, mlp.router, expert_indices)
                
                layer_results = {}
                for expert_idx in expert_indices:
                    pbar.set_postfix({'layer': layer_idx, 'expert': expert_idx})
                    
                    if expert_idx >= len(mlp.scientific_experts):
                        pbar.update(1)
                        continue
                    
                    expert = mlp.scientific_experts[expert_idx]
                    expert_hidden, routing_stats = routed_dict.get(expert_idx, (None, None))
                    
                    if expert_hidden is None:
                        layer_results[f'E{expert_idx}'] = {'skipped': True, 'reason': 'no_routed_tokens'}
                        pbar.update(1)
                        continue
                    
                    flatness = self._compute_expert_flatness(expert, expert_hidden, cached_batches, layer_idx, expert_idx)
                    flatness['routing_stats'] = routing_stats
                    layer_results[f'E{expert_idx}'] = flatness
                    pbar.update(1)
                
                results[f'layer_{layer_idx}'] = layer_results
            
            pbar.close()
        
        finally:
            if temporary_train:
                self._restore_model_state(saved_state)
                self._clear_cuda_cache()
        
        return results

    def _evaluate_dense_ffn_flatness(self, layer_idx: int, hidden_states: torch.Tensor, cached_batches: list) -> dict:
        """Evaluate flatness for dense FFN using fast methods."""
        dense_weights = self.original_dense_ffn[layer_idx]
        mlp = self._get_mlp(layer_idx)
        h = mlp.gate_proj.in_features
        H = mlp.gate_proj.out_features
        
        # Create temporary dense expert
        class DenseFFNExpert(nn.Module):
            def __init__(self, gate_proj, up_proj, down_proj, act_fn):
                super().__init__()
                self.gate_proj = gate_proj
                self.up_proj = up_proj
                self.down_proj = down_proj
                self.act_fn = act_fn
            
            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        dense_gate = nn.Linear(h, H, bias=False, device=self.device, dtype=self.model_dtype)
        dense_up = nn.Linear(h, H, bias=False, device=self.device, dtype=self.model_dtype)
        dense_down = nn.Linear(H, h, bias=False, device=self.device, dtype=self.model_dtype)
        dense_gate.weight.data = dense_weights['gate_proj'].to(self.device)
        dense_up.weight.data = dense_weights['up_proj'].to(self.device)
        dense_down.weight.data = dense_weights['down_proj'].to(self.device)
        
        dense_expert = DenseFFNExpert(dense_gate, dense_up, dense_down, nn.SiLU())
        
        # Use fast flatness if available
        if FAST_FLATNESS_AVAILABLE:
            evaluator = FastFlatnessEvaluator(
                device=self.device,
                method=self.flatness_methods[0] if self.flatness_methods else 'gradient_norm',
                landscape_steps=self.landscape_steps,
                landscape_multiplier=self.landscape_multiplier,
            )
            loss_fn = create_reconstruction_loss_fn()
            
            flatness = {}
            for method in self.flatness_methods:
                try:
                    if method == 'gradient_norm':
                        metrics = evaluator._compute_gradient_norm(dense_expert, hidden_states, loss_fn)
                    elif method == 'landscape':
                        metrics = evaluator._compute_landscape(dense_expert, hidden_states, loss_fn)
                    elif method == 'fisher_trace':
                        metrics = evaluator._compute_fisher_trace(dense_expert, hidden_states, loss_fn)
                    else:
                        metrics = evaluator._compute_gradient_norm(dense_expert, hidden_states, loss_fn)
                    flatness[method] = metrics
                except Exception as e:
                    flatness[method] = {'error': str(e)}
        else:
            flatness = self._compute_expert_flatness(dense_expert, hidden_states, cached_batches, layer_idx, -1)
        
        del dense_expert, dense_gate, dense_up, dense_down
        self._clear_cuda_cache()
        
        return flatness

    def _compute_expert_flatness(
        self, 
        expert: nn.Module, 
        hidden_states: torch.Tensor,
        cached_batches: list,
        layer_idx: int,
        expert_idx: int
    ) -> dict:
        """Compute flatness metrics for an expert."""
        from evaluations.expert_metrics import HessianSpectrum, EpsilonSharpness, HessianDiagonal, create_expert_loss_fn
        
        result = {
            'layer': layer_idx,
            'expert': expert_idx,
            'flatness_method': self.flatness_method,
            'num_samples': len(hidden_states)
        }
        
        # Limit samples
        if len(hidden_states) > self.flatness_max_samples:
            hidden_states = hidden_states[:self.flatness_max_samples]
        
        # Create loss function
        if self.flatness_loss_type == 'final':
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states,
                full_model=self.model,
                model_input_batches=cached_batches,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device=self.device
            )
        else:
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states
            )
        
        # Create dataloader
        expert_dataloader = self._create_expert_dataloader(hidden_states, self.flatness_batch_size)
        
        # Enable gradients
        for param in expert.parameters():
            param.requires_grad = True
        
        try:
            if self.flatness_method == 'hessian':
                # Hessian top eigenvalue
                hessian = HessianSpectrum(expert, loss_fn, self.device, self.power_iterations)
                lambda_max, _ = hessian.compute_top_eigenvalue(expert_dataloader, min(self.metric_num_batches, 10))
                result['hessian_top_eigenvalue'] = float(lambda_max)
                
                # Epsilon sharpness
                eps_sharp = EpsilonSharpness(epsilon=self.sharpness_epsilon)
                result['epsilon_sharpness'] = float(eps_sharp.compute(lambda_max))
                
                # Hessian diagonal/trace
                hess_diag = HessianDiagonal(expert, loss_fn, self.device, self.hutchinson_samples)
                diag_result, _ = hess_diag.compute_trace(expert_dataloader, min(self.metric_num_batches, 10))
                result['hessian_trace'] = float(diag_result['trace'])
        except Exception as e:
            result['error'] = str(e)
            print_rank_0(f"   [Warning] Flatness computation failed: {e}", 0)
        
        return result

    def _temporary_train_experts(self, expert_indices: range, dataloader, num_batches: int):
        """
        Temporarily train specified experts for flatness evaluation.
        
        NOTE: This function is DISABLED due to DeepSpeed ZeRO Stage 2 incompatibility.
        ZeRO Stage 2 hooks into backward() and doesn't support manual gradient updates
        for dynamically added parameters.
        
        Current solution: Evaluate flatness on CURRENT expert state without training.
        This reflects experts' immediate adaptation potential to new data.
        """
        print_rank_0(f"      [SKIP] Temporary training disabled (DeepSpeed ZeRO incompatible)", 0)
        print_rank_0(f"      [INFO] Evaluating flatness on current expert state instead", 0)

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _get_mlp(self, layer_idx):
        """Get MLP module for a layer."""
        if hasattr(self.model, 'module'):
            return self.model.module.model.layers[layer_idx].mlp
        return self.model.model.layers[layer_idx].mlp

    def _get_layers_to_evaluate(self) -> list:
        """Get layer indices to evaluate."""
        if self.metric_all_layers:
            return [i for i, layer in enumerate(self.model.model.layers) 
                    if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0]
        
        # Find the specified MoE layer
        layer_idx = self.metric_layer_idx
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                if layer_idx == 0:
                    return [i]
                layer_idx -= 1
        return []

    def _cache_dataloader_batches(self, dataloader, num_batches: int) -> list:
        """Cache batches from dataloader."""
        cached = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            if 'sources' in batch:
                del batch['sources']
            cached.append({
                'input_ids': batch['input_ids'].cpu(),
                'attention_mask': batch.get('attention_mask', None)
            })
            if cached[-1]['attention_mask'] is not None:
                cached[-1]['attention_mask'] = cached[-1]['attention_mask'].cpu()
        return cached

    def _prepare_hidden_states(self, cached_batches: list, layer_idx: int, max_samples: int = 512) -> Optional[torch.Tensor]:
        """Prepare hidden states for evaluation."""
        self.model.eval()
        hidden_list = []
        total = 0
        
        try:
            with torch.no_grad():
                for batch in cached_batches:
                    if total >= max_samples:
                        break
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    base_model = self.model.module.model if hasattr(self.model, 'module') else self.model.model
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True
                    )
                    
                    hs = outputs.hidden_states[layer_idx]
                    batch_hs = hs.view(-1, hs.size(-1))
                    samples_to_take = min(batch_hs.size(0), max_samples - total)
                    hidden_list.append(batch_hs[:samples_to_take].cpu())
                    total += samples_to_take
                    
                    del outputs, hs, batch_hs
                    self._clear_cuda_cache()
        except Exception as e:
            print_rank_0(f"[Warning] Failed to prepare hidden states: {e}", 0)
            return None
        
        if hidden_list:
            return torch.cat(hidden_list, dim=0).to(self.device)
        return None

    def _compute_dense_output(self, hidden_states: torch.Tensor, dense_weights: dict, mlp) -> torch.Tensor:
        """Compute output using dense FFN weights (moves CPU weights to GPU temporarily)."""
        device = hidden_states.device
        with torch.no_grad():
            gate_w = dense_weights['gate_proj'].to(device)
            up_w = dense_weights['up_proj'].to(device)
            down_w = dense_weights['down_proj'].to(device)
            gate_out = F.silu(F.linear(hidden_states, gate_w))
            up_out = F.linear(hidden_states, up_w)
            output = F.linear(gate_out * up_out, down_w)
        return output

    def _get_all_experts_routed_tokens(self, hidden_states: torch.Tensor, router, expert_range: range, min_tokens: int = 32) -> dict:
        """Get routed tokens for all experts in a range."""
        result = {}
        with torch.no_grad():
            num_tokens, hidden_dim = hidden_states.shape
            x = hidden_states.unsqueeze(0)
            routing_weights, selected_experts = router(x)
            
            for expert_idx in expert_range:
                expert_mask = (selected_experts == expert_idx).any(dim=-1)
                routed_indices = expert_mask.nonzero(as_tuple=True)[0]
                num_routed = len(routed_indices)
                
                if num_routed == 0:
                    result[expert_idx] = (None, {'num_routed': 0, 'percentage': 0.0})
                    continue
                
                routed_tokens = hidden_states[routed_indices]
                if num_routed < min_tokens:
                    repeat = (min_tokens // num_routed) + 1
                    routed_tokens = routed_tokens.repeat(repeat, 1)[:min_tokens]
                
                result[expert_idx] = (routed_tokens, {
                    'num_routed': num_routed,
                    'percentage': 100.0 * num_routed / num_tokens
                })
        
        return result

    def _create_expert_dataloader(self, hidden_states: torch.Tensor, batch_size: int):
        """Create simple dataloader from hidden states."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        return torch.utils.data.DataLoader(SimpleDataset(hidden_states), batch_size=batch_size, shuffle=False)

    def _save_model_state(self) -> dict:
        """Save current model state for restoration."""
        state = {}
        for i, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            if hasattr(mlp, 'scientific_experts') and len(mlp.scientific_experts) > 0:
                layer_state = {'experts': {}, 'router': {}}
                for idx, expert in enumerate(mlp.scientific_experts):
                    layer_state['experts'][idx] = {name: param.data.clone() for name, param in expert.named_parameters()}
                if hasattr(mlp.router, 'classifier'):
                    layer_state['router'] = {name: param.data.clone() for name, param in mlp.router.classifier.named_parameters()}
                state[i] = layer_state
        return state

    def _restore_model_state(self, state: dict):
        """Restore model state from saved state."""
        for i, layer in enumerate(self.model.model.layers):
            if i not in state:
                continue
            mlp = layer.mlp
            layer_state = state[i]
            for idx, expert_state in layer_state['experts'].items():
                if idx < len(mlp.scientific_experts):
                    for name, saved in expert_state.items():
                        dict(mlp.scientific_experts[idx].named_parameters())[name].data.copy_(saved)
            if 'router' in layer_state and hasattr(mlp.router, 'classifier'):
                for name, saved in layer_state['router'].items():
                    dict(mlp.router.classifier.named_parameters())[name].data.copy_(saved)

    def _clear_cuda_cache(self):
        """Clear CUDA cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _save_metric_results(self, i_task: int, checkpoint_name: str, results: dict):
        """Save metric results to JSON file."""
        if not hasattr(self.args, 'output_dir') or self.args.output_dir is None:
            return
        
        subfolder = 'with_expansion' if self.task0_expansion_mode else 'without_expansion'
        metrics_dir = os.path.join(self.args.output_dir, 'expert_metrics', subfolder)
        os.makedirs(metrics_dir, exist_ok=True)
        
        filename = f"metrics_task{i_task}_{checkpoint_name}.json"
        filepath = os.path.join(metrics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_rank_0(f"[Metrics] Saved to {filepath}", 0)

    # =========================================================================
    # Inference Methods
    # =========================================================================
    
    def evaluate(self, round, infer_task_id, task):
        """Evaluate on a specific task."""
        self.evaluate_one_task(round, infer_task_id, task)

    def evaluate_one_task(self, round, infer_task_id, task):
        """Evaluate on one task and save results."""
        import torch.distributed as dist
        
        infer_dataloader = self.test_task_list[task]
        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        predicted_sequences = []
        sources_sequences = []
        label_sequences = []
        self.model.eval()

        for step, batch in enumerate(infer_dataloader):
            ground_truths_ids = self.tokenizer(
                batch['gts'], 
                truncation=True,
                max_length=self.args.max_ans_len,
                add_special_tokens=False,
                padding='max_length',
                return_tensors='pt'
            )['input_ids'].to(self.device)
            del batch['gts']
            del batch['sources']
            batch = to_device(batch, self.device)
            
            if self.args.global_rank == 0:
                progress_bar.update(1)

            with torch.no_grad():
                generate_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=self.args.max_ans_len,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.unk_token_id,
                    generation_config=generation_config,
                    use_cache=False
                )

            gathered_ids, _ = self._dist_results_gather(generate_ids, self.tokenizer.eos_token_id)
            gathered_labels, _ = self._dist_results_gather(ground_truths_ids, self.tokenizer.eos_token_id)

            if self.args.global_rank <= 0:
                input_len = batch['input_ids'].shape[1]
                sources_sequences.extend(self.tokenizer.batch_decode(
                    gathered_ids[:, :input_len], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                predicted_sequences.extend(self.tokenizer.batch_decode(
                    gathered_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                label_sequences.extend(self.tokenizer.batch_decode(
                    gathered_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        if self.args.global_rank <= 0:
            evaluators = _get_task_evaluators()
            eval_fn = evaluators.get(task, lambda p, g: {})
            evaluation_result = eval_fn(predicted_sequences, label_sequences)
            
            print_rank_0(f"Evaluation result for {task}: {evaluation_result}", self.args.global_rank)
            
            # Save results
            df = {"eval": evaluation_result, 'prompts': sources_sequences, 
                  'results': predicted_sequences, 'labels': label_sequences}
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, f"results-{round}-{infer_task_id}-{task}.json"), "w+", encoding='utf-8') as f:
                json.dump(df, f, ensure_ascii=False, indent=4)

    def _dist_results_gather(self, tensor, pad_value):
        """Gather distributed results across all processes."""
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        
        max_size = max(s.item() for s in all_sizes)
        
        if tensor.shape[0] < max_size:
            padding = torch.full(
                (max_size - tensor.shape[0], tensor.shape[1]),
                pad_value, dtype=tensor.dtype, device=tensor.device
            )
            tensor = torch.cat([tensor, padding], dim=0)
        
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        result = [t[:all_sizes[i].item()] for i, t in enumerate(gathered)]
        return torch.cat(result, dim=0), max_size
