"""
MoE Checkpoint Manager for CKA-Guided Upcycling.

Manages saving and loading of complete training state including:
- MoE architecture configuration (expert ranges, routing config)
- Model weights
- CKA baseline outputs
- Replay buffer data
- Training state (current task, completed tasks, history)
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import torch
from torch import Tensor

from utils.utils import print_rank_0


class MoECheckpointManager:
    """
    Manages MoE checkpoint save/load with full state preservation.
    
    Checkpoint Structure:
        checkpoint_dir/
        ├── moe_config.json          # MoE architecture config
        ├── cka_baseline.pt          # CKA baseline outputs (CPU tensors)
        ├── cka_baseline_inputs.pt   # CKA baseline input batches
        ├── replay_buffer.pt         # Replay buffer data
        ├── training_state.json      # Training progress state
        └── model files...           # Standard HF model files
    """
    
    @staticmethod
    def get_moe_config(model, trainer) -> Dict[str, Any]:
        """
        Extract current MoE configuration from model and trainer.
        
        Args:
            model: The MoE model
            trainer: The Upcycle trainer instance
            
        Returns:
            Dictionary containing MoE configuration
        """
        # Get model layers to find MoE structure
        if hasattr(model, 'module'):
            model_unwrapped = model.module
        else:
            model_unwrapped = model
            
        # Extract expert configuration from first MoE layer
        num_experts = 0
        num_layers = 0
        
        if hasattr(model_unwrapped, 'model') and hasattr(model_unwrapped.model, 'layers'):
            num_layers = len(model_unwrapped.model.layers)
            layer0 = model_unwrapped.model.layers[0]
            if hasattr(layer0.mlp, 'scientific_experts'):
                num_experts = len(layer0.mlp.scientific_experts)
        
        # Extract trainer configuration
        moe_config = {
            'num_layers': num_layers,
            'total_num_experts': num_experts,
            'num_experts_per_task': getattr(trainer, 'num_experts_per_task', 8),
            'num_activated_experts': getattr(trainer, 'num_activated_experts', 2),
            'task2expert_range': {},
            'upcycle_interval': getattr(trainer, 'upcycle_interval', 1),
            'non_expansion_mode': getattr(trainer, 'non_expansion_mode', True),
            'router_init_method': getattr(trainer.args, 'router_init_method', 'zero_bias'),
        }
        
        # Convert task2expert_range (ranges are not JSON serializable)
        if hasattr(trainer, 'task2expert_range'):
            for task_id, expert_range in trainer.task2expert_range.items():
                moe_config['task2expert_range'][str(task_id)] = list(expert_range)
        
        return moe_config
    
    @staticmethod
    def save_moe_config(checkpoint_dir: str, moe_config: Dict[str, Any]) -> None:
        """Save MoE configuration to JSON file."""
        config_path = os.path.join(checkpoint_dir, 'moe_config.json')
        with open(config_path, 'w') as f:
            json.dump(moe_config, f, indent=2)
        print_rank_0(f"[Checkpoint] Saved MoE config to {config_path}", 0)
    
    @staticmethod
    def load_moe_config(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """Load MoE configuration from JSON file."""
        config_path = os.path.join(checkpoint_dir, 'moe_config.json')
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Convert task2expert_range keys back to int
        if 'task2expert_range' in config:
            config['task2expert_range'] = {
                int(k): range(v[0], v[1]) if len(v) == 2 else range(v[0], v[-1]+1)
                for k, v in config['task2expert_range'].items()
            }
        return config
    
    @staticmethod
    def save_cka_baseline(checkpoint_dir: str, 
                          baseline_outputs: Dict[int, Tensor],
                          baseline_inputs: List[Dict] = None) -> None:
        """
        Save CKA baseline outputs and inputs.
        
        Args:
            checkpoint_dir: Directory to save to
            baseline_outputs: Dict mapping layer_idx -> cached output tensor
            baseline_inputs: List of input batch dicts used for baseline
        """
        # Save baseline outputs (move to CPU if needed)
        baseline_path = os.path.join(checkpoint_dir, 'cka_baseline.pt')
        cpu_outputs = {}
        for layer_idx, tensor in baseline_outputs.items():
            if isinstance(tensor, Tensor):
                cpu_outputs[layer_idx] = tensor.cpu()
            else:
                cpu_outputs[layer_idx] = tensor
        torch.save(cpu_outputs, baseline_path)
        
        # Save baseline inputs if provided
        if baseline_inputs:
            inputs_path = os.path.join(checkpoint_dir, 'cka_baseline_inputs.pt')
            # Ensure all tensors are on CPU
            cpu_inputs = []
            for batch in baseline_inputs:
                cpu_batch = {}
                for k, v in batch.items():
                    if isinstance(v, Tensor):
                        cpu_batch[k] = v.cpu()
                    else:
                        cpu_batch[k] = v
                cpu_inputs.append(cpu_batch)
            torch.save(cpu_inputs, inputs_path)
        
        print_rank_0(f"[Checkpoint] Saved CKA baseline ({len(cpu_outputs)} layers)", 0)
    
    @staticmethod
    def load_cka_baseline(checkpoint_dir: str) -> Tuple[Dict[int, Tensor], List[Dict]]:
        """
        Load CKA baseline outputs and inputs.
        
        Returns:
            Tuple of (baseline_outputs dict, baseline_inputs list)
            Returns empty dict/list if files don't exist
        """
        baseline_outputs = {}
        baseline_inputs = []
        
        baseline_path = os.path.join(checkpoint_dir, 'cka_baseline.pt')
        if os.path.exists(baseline_path):
            baseline_outputs = torch.load(baseline_path, map_location='cpu')
            print_rank_0(f"[Checkpoint] Loaded CKA baseline ({len(baseline_outputs)} layers)", 0)
        
        inputs_path = os.path.join(checkpoint_dir, 'cka_baseline_inputs.pt')
        if os.path.exists(inputs_path):
            baseline_inputs = torch.load(inputs_path, map_location='cpu')
            print_rank_0(f"[Checkpoint] Loaded CKA baseline inputs ({len(baseline_inputs)} batches)", 0)
        
        return baseline_outputs, baseline_inputs
    
    @staticmethod
    def save_replay_buffer(checkpoint_dir: str, replay_buffer) -> None:
        """
        Save replay buffer data.
        
        Args:
            checkpoint_dir: Directory to save to
            replay_buffer: ReplayBuffer instance
        """
        buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.pt')
        
        # Get buffer data and ensure tensors are on CPU
        buffer_data = {}
        for task_id, batches in replay_buffer.buffers.items():
            cpu_batches = []
            for batch in batches:
                cpu_batch = {}
                for k, v in batch.items():
                    if isinstance(v, Tensor):
                        cpu_batch[k] = v.cpu()
                    else:
                        cpu_batch[k] = v
                cpu_batches.append(cpu_batch)
            buffer_data[task_id] = cpu_batches
        
        torch.save(buffer_data, buffer_path)
        total_batches = sum(len(b) for b in buffer_data.values())
        print_rank_0(f"[Checkpoint] Saved replay buffer ({total_batches} batches across {len(buffer_data)} tasks)", 0)
    
    @staticmethod
    def load_replay_buffer(checkpoint_dir: str, replay_buffer) -> bool:
        """
        Load replay buffer data into existing buffer.
        
        Args:
            checkpoint_dir: Directory to load from
            replay_buffer: ReplayBuffer instance to populate
            
        Returns:
            True if loaded successfully, False otherwise
        """
        buffer_path = os.path.join(checkpoint_dir, 'replay_buffer.pt')
        if not os.path.exists(buffer_path):
            return False
        
        buffer_data = torch.load(buffer_path, map_location='cpu')
        replay_buffer.buffers = buffer_data
        total_batches = sum(len(b) for b in buffer_data.values())
        print_rank_0(f"[Checkpoint] Loaded replay buffer ({total_batches} batches across {len(buffer_data)} tasks)", 0)
        return True
    
    @staticmethod
    def save_training_state(checkpoint_dir: str, 
                           current_task_id: int,
                           completed_tasks: List[str],
                           cka_history: Dict = None,
                           extra_state: Dict = None) -> None:
        """
        Save training progress state.
        
        Args:
            checkpoint_dir: Directory to save to
            current_task_id: Current task index
            completed_tasks: List of completed task names
            cka_history: Optional dict of CKA scores history
            extra_state: Optional additional state to save
        """
        state_path = os.path.join(checkpoint_dir, 'training_state.json')
        
        state = {
            'current_task_id': current_task_id,
            'completed_tasks': completed_tasks,
            'cka_history': cka_history or {},
        }
        if extra_state:
            state.update(extra_state)
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        print_rank_0(f"[Checkpoint] Saved training state (task {current_task_id})", 0)
    
    @staticmethod
    def load_training_state(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """Load training progress state."""
        state_path = os.path.join(checkpoint_dir, 'training_state.json')
        if not os.path.exists(state_path):
            return None
        with open(state_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_full_checkpoint(checkpoint_dir: str,
                            task_id: int,
                            model,
                            trainer,
                            baseline_outputs: Dict[int, Tensor],
                            baseline_inputs: List[Dict],
                            replay_buffer,
                            completed_tasks: List[str],
                            cka_history: Dict = None,
                            rank: int = 0) -> None:
        """
        Save complete checkpoint with all state.
        
        Args:
            checkpoint_dir: Base directory (will save to checkpoint_dir/task_id/)
            task_id: Current task ID
            model: The model to save
            trainer: The trainer instance
            baseline_outputs: CKA baseline outputs
            baseline_inputs: CKA baseline input batches
            replay_buffer: Replay buffer instance
            completed_tasks: List of completed task names
            cka_history: CKA history dict
            rank: Process rank (only rank 0 saves)
        """
        if rank != 0:
            return
        
        task_checkpoint_dir = os.path.join(checkpoint_dir, str(task_id))
        os.makedirs(task_checkpoint_dir, exist_ok=True)
        
        print_rank_0(f"[Checkpoint] Saving full checkpoint to {task_checkpoint_dir}", rank)
        
        # 1. Save MoE config
        moe_config = MoECheckpointManager.get_moe_config(model, trainer)
        MoECheckpointManager.save_moe_config(task_checkpoint_dir, moe_config)
        
        # 2. Save CKA baseline
        MoECheckpointManager.save_cka_baseline(
            task_checkpoint_dir, baseline_outputs, baseline_inputs
        )
        
        # 3. Save replay buffer
        MoECheckpointManager.save_replay_buffer(task_checkpoint_dir, replay_buffer)
        
        # 4. Save training state
        MoECheckpointManager.save_training_state(
            task_checkpoint_dir,
            current_task_id=task_id,
            completed_tasks=completed_tasks,
            cka_history=cka_history
        )
        
        print_rank_0(f"[Checkpoint] Full checkpoint saved for task {task_id}", rank)
    
    @staticmethod
    def load_full_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
        """
        Load complete checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory (e.g., output/3/)
            
        Returns:
            Dictionary containing:
                - moe_config: MoE architecture config
                - baseline_outputs: CKA baseline tensors
                - baseline_inputs: CKA baseline input batches
                - training_state: Training progress state
        """
        result = {
            'moe_config': None,
            'baseline_outputs': {},
            'baseline_inputs': [],
            'training_state': None,
            'checkpoint_dir': checkpoint_dir,
        }
        
        # Load MoE config
        result['moe_config'] = MoECheckpointManager.load_moe_config(checkpoint_dir)
        
        # Load CKA baseline
        baseline_outputs, baseline_inputs = MoECheckpointManager.load_cka_baseline(checkpoint_dir)
        result['baseline_outputs'] = baseline_outputs
        result['baseline_inputs'] = baseline_inputs
        
        # Load training state
        result['training_state'] = MoECheckpointManager.load_training_state(checkpoint_dir)
        
        return result
    
    @staticmethod
    def rebuild_moe_structure(model, trainer, moe_config: Dict[str, Any]) -> None:
        """
        Rebuild MoE structure to match checkpoint configuration.
        
        This method upcycles the model to have the correct number of experts
        based on the saved configuration.
        
        Args:
            model: Base model (may be vanilla or partially upcycled)
            trainer: Upcycle trainer instance
            moe_config: Loaded MoE configuration
        """
        print_rank_0(f"[Checkpoint] Rebuilding MoE structure with config: "
                    f"{moe_config.get('total_num_experts', 0)} experts", 0)
        
        # Restore task2expert_range
        if 'task2expert_range' in moe_config:
            trainer.task2expert_range = moe_config['task2expert_range']
        
        # Restore other config
        if 'num_experts_per_task' in moe_config:
            trainer.num_experts_per_task = moe_config['num_experts_per_task']
        if 'num_activated_experts' in moe_config:
            trainer.num_activated_experts = moe_config['num_activated_experts']
        
        # The actual MoE structure will be rebuilt by upcycling
        # For each task in task2expert_range, we need to call upcycle
        num_completed_tasks = len(moe_config.get('task2expert_range', {}))
        
        print_rank_0(f"[Checkpoint] MoE config restored: {num_completed_tasks} tasks, "
                    f"expert ranges: {moe_config.get('task2expert_range', {})}", 0)


def fix_config_json(checkpoint_dir: str) -> bool:
    """
    Fix common issues in config.json (e.g., token_id being list instead of int).
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        True if fixes were applied, False otherwise
    """
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    fixed = False
    for key in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
        if isinstance(config.get(key), list):
            config[key] = config[key][0] if config[key] else None
            fixed = True
    
    if fixed:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_rank_0(f"[Checkpoint] Fixed config.json token IDs", 0)
    
    return fixed
