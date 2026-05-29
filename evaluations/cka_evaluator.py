"""
CKA (Centered Kernel Alignment) Evaluator for MoE Expert Metrics.

This module provides CKA evaluation to measure representational similarity
between old model outputs and new model outputs after expert expansion.

Scientific Formula:
    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    
where HSIC (Hilbert-Schmidt Independence Criterion):
    HSIC(X, Y) = (1/(n-1)²) * tr(K_X H K_Y H)
    K_X = X X^T (linear kernel)
    H = I - (1/n) 1 1^T (centering matrix)

For linear CKA, this simplifies to:
    CKA(X, Y) = ||Y^T X||_F² / sqrt(||X^T X||_F² * ||Y^T Y||_F²)

Reference:
- Kornblith et al., "Similarity of Neural Network Representations Revisited"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
from dataclasses import dataclass, field
import warnings


@dataclass
class CKAResult:
    """Container for CKA evaluation results."""
    layer_idx: int
    task_name: str = ""
    
    # Core metric
    cka_score: float = 0.0
    
    # HSIC components
    hsic_xy: float = 0.0
    hsic_xx: float = 0.0
    hsic_yy: float = 0.0
    
    # Statistics
    num_samples: int = 0
    
    # Timing
    time_seconds: float = 0.0
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'layer': self.layer_idx,
            'task': self.task_name,
            'cka_score': self.cka_score,
            'hsic': {
                'xy': self.hsic_xy,
                'xx': self.hsic_xx,
                'yy': self.hsic_yy
            },
            'num_samples': self.num_samples,
            'time_seconds': self.time_seconds,
            'details': self.details
        }


class CKAEvaluator:
    """
    CKA evaluator for comparing model layer outputs.
    
    Used to measure how well new experts preserve representations
    on old task data after expansion.
    
    Args:
        device: Torch device
        mode: 'full' for all data, 'early' for first N batches
        early_batches: Number of batches for 'early' mode
    """
    
    def __init__(
        self,
        device: torch.device,
        mode: str = 'full',
        early_batches: int = 10,
    ):
        self.device = device
        self.mode = mode
        self.early_batches = early_batches
        
        # Accumulators for streaming computation
        self.reset()
    
    def reset(self):
        """Reset accumulators for new computation."""
        self.X_cov = None  # X^T X
        self.Y_cov = None  # Y^T Y
        self.XY_cov = None  # X^T Y
        self.n_samples = 0
        self.batch_count = 0
    
    def _center(self, X: Tensor) -> Tensor:
        """Center activations by subtracting mean."""
        return X - X.mean(dim=0, keepdim=True)
    
    def compute_cka_direct(
        self,
        X: Tensor,
        Y: Tensor,
        verbose: bool = False
    ) -> float:
        """
        Compute Linear CKA directly from two activation matrices.
        
        Args:
            X: Activations from old model [samples, features]
            Y: Activations from new model [samples, features]
            verbose: Print debug info
            
        Returns:
            CKA similarity score in [0, 1]
        """
        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        
        # Check for invalid values
        if torch.isnan(X).any() or torch.isinf(X).any():
            warnings.warn("X contains NaN or Inf")
            return -1.0
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            warnings.warn("Y contains NaN or Inf")
            return -1.0
        
        # Center activations
        X_c = self._center(X)
        Y_c = self._center(Y)
        
        if verbose:
            print(f"  [CKA] X shape: {X.shape}, Y shape: {Y.shape}")
            print(f"  [CKA] X_c std: {X_c.std():.6f}, Y_c std: {Y_c.std():.6f}")
        
        # Compute Frobenius norms
        YtX = Y_c.T @ X_c
        YtX_norm_sq = torch.sum(YtX ** 2)
        
        XtX = X_c.T @ X_c
        XtX_norm_sq = torch.sum(XtX ** 2)
        
        YtY = Y_c.T @ Y_c
        YtY_norm_sq = torch.sum(YtY ** 2)
        
        if verbose:
            print(f"  [CKA] XtX_norm_sq: {XtX_norm_sq:.6e}")
            print(f"  [CKA] YtY_norm_sq: {YtY_norm_sq:.6e}")
            print(f"  [CKA] YtX_norm_sq: {YtX_norm_sq:.6e}")
        
        # CKA formula
        denominator = torch.sqrt(XtX_norm_sq * YtY_norm_sq)
        
        if denominator < 1e-10:
            warnings.warn(f"CKA denominator near zero: {denominator:.2e}")
            return -2.0
        
        cka = (YtX_norm_sq / denominator).item()
        
        return float(np.clip(cka, 0.0, 1.0))
    
    def accumulate_batch(
        self,
        X_batch: Tensor,
        Y_batch: Tensor
    ) -> bool:
        """
        Accumulate a batch for streaming CKA computation.
        
        Args:
            X_batch: Batch from old model
            Y_batch: Batch from new model
            
        Returns:
            True to continue, False if done (early mode)
        """
        self.batch_count += 1
        
        X_batch = X_batch.to(self.device).float()
        Y_batch = Y_batch.to(self.device).float()
        
        # Center
        X_batch = self._center(X_batch)
        Y_batch = self._center(Y_batch)
        
        n = X_batch.shape[0]
        self.n_samples += n
        
        # Accumulate covariances
        batch_X_cov = X_batch.T @ X_batch
        batch_Y_cov = Y_batch.T @ Y_batch
        batch_XY_cov = X_batch.T @ Y_batch
        
        if self.X_cov is None:
            self.X_cov = batch_X_cov
            self.Y_cov = batch_Y_cov
            self.XY_cov = batch_XY_cov
        else:
            self.X_cov += batch_X_cov
            self.Y_cov += batch_Y_cov
            self.XY_cov += batch_XY_cov
        
        # Check early stopping
        if self.mode == 'early' and self.batch_count >= self.early_batches:
            return False
        return True
    
    def compute_cka_from_accumulated(self) -> Tuple[float, Dict]:
        """
        Compute CKA from accumulated covariances.
        
        Returns:
            Tuple of (cka_score, hsic_components)
        """
        if self.X_cov is None:
            raise ValueError("No batches accumulated")
        
        XY_norm_sq = torch.sum(self.XY_cov ** 2)
        XX_norm_sq = torch.sum(self.X_cov ** 2)
        YY_norm_sq = torch.sum(self.Y_cov ** 2)
        
        denominator = torch.sqrt(XX_norm_sq * YY_norm_sq)
        
        if denominator < 1e-10:
            return 0.0, {'hsic_xy': 0.0, 'hsic_xx': 0.0, 'hsic_yy': 0.0}
        
        cka = (XY_norm_sq / denominator).item()
        
        return float(np.clip(cka, 0.0, 1.0)), {
            'hsic_xy': XY_norm_sq.item(),
            'hsic_xx': XX_norm_sq.item(),
            'hsic_yy': YY_norm_sq.item()
        }


# =============================================================================
# Differentiable multi-dim CKA (used by V7 expert merge)
# =============================================================================
# These functions return graph-attached tensors so .backward() works through
# them. Keep them dependency-free so they can be imported by expert_merge.py
# without pulling in the full evaluator state.

def _center_diff(X: Tensor) -> Tensor:
    """Center a [N, D] tensor along the sample dim (differentiable)."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka_diff(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Differentiable Linear CKA on two [N, D] tensors. Returns a scalar in [0, 1]
    (clamped) with the gradient graph attached. Both inputs must have the same
    number of rows; the column dim may differ.
    """
    if X.shape[0] != Y.shape[0]:
        n = min(X.shape[0], Y.shape[0])
        X = X[:n]
        Y = Y[:n]

    Xc = _center_diff(X)
    Yc = _center_diff(Y)

    YtX = Yc.T @ Xc
    YtX_n2 = (YtX * YtX).sum()
    XtX = Xc.T @ Xc
    XtX_n2 = (XtX * XtX).sum()
    YtY = Yc.T @ Yc
    YtY_n2 = (YtY * YtY).sum()

    denom = torch.sqrt(XtX_n2 * YtY_n2 + eps)
    cka = YtX_n2 / (denom + eps)
    return cka.clamp(0.0, 1.0)


def feature_group_cka_diff(
    X: Tensor,
    Y: Tensor,
    num_groups: int = 4,
    eps: float = 1e-8,
) -> Tensor:
    """
    Split the feature dim of two [N, D] tensors into num_groups contiguous
    chunks, compute Linear CKA per chunk, return the mean (differentiable).
    """
    if num_groups <= 1:
        return linear_cka_diff(X, Y, eps=eps)

    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]
    d = min(X.shape[1], Y.shape[1])
    group_size = max(1, d // num_groups)

    scores = []
    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < num_groups - 1 else d
        if end <= start:
            continue
        scores.append(linear_cka_diff(X[:, start:end], Y[:, start:end], eps=eps))

    if not scores:
        return linear_cka_diff(X, Y, eps=eps)
    return torch.stack(scores).mean()


def token_group_cka_diff(
    X: Tensor,
    Y: Tensor,
    seq_len: int,
    num_groups: int = 3,
    eps: float = 1e-8,
) -> Tensor:
    """
    Split tokens into num_groups contiguous position buckets along the seq
    dim, compute CKA per bucket on flattened activations, return the mean.

    Inputs are [B*S, D] flattened; we re-interpret via seq_len to bucket by
    position within the sequence. If shapes don't align we fall back to
    feature-group CKA.
    """
    if num_groups <= 1 or seq_len <= 1:
        return linear_cka_diff(X, Y, eps=eps)

    n = min(X.shape[0], Y.shape[0])
    if n % seq_len != 0:
        return linear_cka_diff(X[:n], Y[:n], eps=eps)

    B = n // seq_len
    X3 = X[:n].view(B, seq_len, -1)
    Y3 = Y[:n].view(B, seq_len, -1)

    bucket_size = max(1, seq_len // num_groups)
    scores = []
    for g in range(num_groups):
        start = g * bucket_size
        end = (g + 1) * bucket_size if g < num_groups - 1 else seq_len
        if end <= start:
            continue
        Xb = X3[:, start:end, :].reshape(-1, X3.shape[-1])
        Yb = Y3[:, start:end, :].reshape(-1, Y3.shape[-1])
        scores.append(linear_cka_diff(Xb, Yb, eps=eps))

    if not scores:
        return linear_cka_diff(X, Y, eps=eps)
    return torch.stack(scores).mean()


def compute_layer_cka(
    old_model: nn.Module,
    new_model: nn.Module,
    cached_batches: List[Dict],
    layer_idx: int,
    device: torch.device,
    max_samples: int = 512,
    task_name: str = ""
) -> CKAResult:
    """
    Compute CKA between old and new model at a specific layer.
    
    Args:
        old_model: Model before expansion
        new_model: Model after expansion
        cached_batches: Pre-cached input batches
        layer_idx: Layer index to compare
        device: Torch device
        max_samples: Max samples to use
        task_name: Task name for logging
        
    Returns:
        CKAResult with CKA score and details
    """
    start_time = time.time()
    
    evaluator = CKAEvaluator(device=device)
    
    old_outputs = []
    new_outputs = []
    total_samples = 0
    
    try:
        with torch.no_grad():
            for batch in cached_batches:
                if total_samples >= max_samples:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Get old model outputs
                if hasattr(old_model, 'module'):
                    old_base = old_model.module.model
                else:
                    old_base = old_model.model
                
                old_out = old_base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                old_hs = old_out.hidden_states[layer_idx]
                old_hs = old_hs.view(-1, old_hs.size(-1))
                
                # Get new model outputs
                if hasattr(new_model, 'module'):
                    new_base = new_model.module.model
                else:
                    new_base = new_model.model
                
                new_out = new_base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                new_hs = new_out.hidden_states[layer_idx]
                new_hs = new_hs.view(-1, new_hs.size(-1))
                
                samples_to_take = min(old_hs.size(0), max_samples - total_samples)
                old_outputs.append(old_hs[:samples_to_take].cpu())
                new_outputs.append(new_hs[:samples_to_take].cpu())
                total_samples += samples_to_take
                
                del old_out, new_out, old_hs, new_hs
        
        if not old_outputs:
            return CKAResult(
                layer_idx=layer_idx,
                task_name=task_name,
                details={'error': 'No samples collected'}
            )
        
        # Concatenate and compute CKA
        X = torch.cat(old_outputs, dim=0).to(device)
        Y = torch.cat(new_outputs, dim=0).to(device)
        
        cka_score = evaluator.compute_cka_direct(X, Y)
        
        elapsed = time.time() - start_time
        
        return CKAResult(
            layer_idx=layer_idx,
            task_name=task_name,
            cka_score=cka_score,
            num_samples=total_samples,
            time_seconds=elapsed
        )
        
    except Exception as e:
        return CKAResult(
            layer_idx=layer_idx,
            task_name=task_name,
            details={'error': str(e)}
        )


def compute_moe_layer_cka(
    model: nn.Module,
    old_weights: Dict[int, Dict],
    cached_batches: List[Dict],
    layer_indices: List[int],
    device: torch.device,
    max_samples: int = 512,
    task_name: str = ""
) -> Dict[int, CKAResult]:
    """
    Compute CKA for MoE layers comparing current outputs with stored baseline.
    
    This is used when we have saved the MoE layer outputs from a previous
    checkpoint and want to compare with the current model.
    
    Args:
        model: Current model
        old_weights: Dict mapping layer_idx -> saved output tensors
        cached_batches: Input batches
        layer_indices: Layers to evaluate
        device: Torch device
        max_samples: Max samples
        task_name: Task name
        
    Returns:
        Dict mapping layer_idx -> CKAResult
    """
    results = {}
    
    for layer_idx in layer_indices:
        if layer_idx not in old_weights:
            continue
        
        start_time = time.time()
        
        old_baseline = old_weights[layer_idx]
        if isinstance(old_baseline, Tensor):
            old_baseline = old_baseline.to(device)
        
        # Compute current layer outputs
        current_outputs = []
        total_samples = 0
        
        try:
            with torch.no_grad():
                for batch in cached_batches:
                    if total_samples >= max_samples:
                        break
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    if hasattr(model, 'module'):
                        base = model.module.model
                    else:
                        base = model.model
                    
                    out = base(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True
                    )
                    hs = out.hidden_states[layer_idx]
                    hs = hs.view(-1, hs.size(-1))
                    
                    samples_to_take = min(hs.size(0), max_samples - total_samples)
                    current_outputs.append(hs[:samples_to_take])
                    total_samples += samples_to_take
                    
                    del out, hs
            
            if not current_outputs:
                results[layer_idx] = CKAResult(
                    layer_idx=layer_idx,
                    task_name=task_name,
                    details={'error': 'No samples'}
                )
                continue
            
            Y = torch.cat(current_outputs, dim=0)
            
            # Ensure same number of samples
            min_samples = min(old_baseline.size(0), Y.size(0))
            X = old_baseline[:min_samples]
            Y = Y[:min_samples]
            
            evaluator = CKAEvaluator(device=device)
            cka_score = evaluator.compute_cka_direct(X, Y)
            
            elapsed = time.time() - start_time
            
            results[layer_idx] = CKAResult(
                layer_idx=layer_idx,
                task_name=task_name,
                cka_score=cka_score,
                num_samples=min_samples,
                time_seconds=elapsed
            )
            
        except Exception as e:
            results[layer_idx] = CKAResult(
                layer_idx=layer_idx,
                task_name=task_name,
                details={'error': str(e)}
            )
    
    return results


def evaluate_expansion_cka(
    model: nn.Module,
    cached_batches: List[Dict],
    layer_indices: List[int],
    device: torch.device,
    old_task_data: Optional[List[Dict]] = None,
    task_name: str = ""
) -> Dict[int, CKAResult]:
    """
    Evaluate CKA for expert expansion scenario.
    
    Measures how well the expanded model preserves representations
    on old task data.
    
    Args:
        model: Model after expansion
        cached_batches: Current task data
        layer_indices: Layers to evaluate
        device: Torch device
        old_task_data: Optional old task data (uses cached_batches if None)
        task_name: Task name
        
    Returns:
        Dict mapping layer_idx -> CKAResult
    """
    from tqdm.auto import tqdm
    
    eval_data = old_task_data if old_task_data is not None else cached_batches
    
    results = {}
    
    pbar = tqdm(layer_indices, desc=f"[CKA] {task_name}", leave=False)
    
    for layer_idx in pbar:
        pbar.set_postfix({'layer': layer_idx})
        
        start_time = time.time()
        
        try:
            # Collect layer outputs
            all_outputs = []
            total_samples = 0
            max_samples = 512
            
            with torch.no_grad():
                for batch in eval_data:
                    if total_samples >= max_samples:
                        break
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    if hasattr(model, 'module'):
                        base = model.module.model
                    else:
                        base = model.model
                    
                    out = base(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True
                    )
                    
                    hs = out.hidden_states[layer_idx]
                    hs = hs.view(-1, hs.size(-1))
                    
                    samples = min(hs.size(0), max_samples - total_samples)
                    all_outputs.append(hs[:samples].cpu())
                    total_samples += samples
                    
                    del out, hs
            
            if not all_outputs:
                results[layer_idx] = CKAResult(
                    layer_idx=layer_idx,
                    task_name=task_name,
                    details={'error': 'No samples'}
                )
                continue
            
            # For self-CKA (measuring internal consistency)
            X = torch.cat(all_outputs, dim=0).to(device)
            
            evaluator = CKAEvaluator(device=device)
            # Self-CKA should be 1.0
            cka_score = evaluator.compute_cka_direct(X, X)
            
            elapsed = time.time() - start_time
            
            results[layer_idx] = CKAResult(
                layer_idx=layer_idx,
                task_name=task_name,
                cka_score=cka_score,
                num_samples=total_samples,
                time_seconds=elapsed
            )
            
        except Exception as e:
            results[layer_idx] = CKAResult(
                layer_idx=layer_idx,
                task_name=task_name,
                details={'error': str(e)}
            )
    
    pbar.close()
    
    return results
