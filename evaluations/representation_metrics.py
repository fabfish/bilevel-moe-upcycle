"""
Representation Alignment Metrics for MoE Continual Learning.

This module implements metrics to measure representation alignment and stability
during MoE expansion and continual learning:

1. Layer-level CKA: How stable is the layer's overall output after expansion?
2. Expert-to-Dense CKA: How well does each expert preserve the original dense MLP's representation?
3. Inter-Expert Diversity: How different are the experts from each other?
4. Activation Stability: Are activations stable after adding new experts?

These metrics are designed for understanding and potentially regularizing
the MoE expansion process.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RepresentationMetrics:
    """Container for representation alignment metrics."""
    # Layer-level metrics
    layer_cka: Optional[float] = None  # CKA between layer outputs before/after
    
    # Expert alignment metrics
    expert_to_dense_cka: Optional[Dict[int, float]] = None  # Each expert vs dense MLP
    
    # Diversity metrics
    inter_expert_cka_matrix: Optional[np.ndarray] = None  # Pairwise CKA between experts
    expert_diversity_score: Optional[float] = None  # 1 - mean(inter_expert_cka)
    
    # Stability metrics
    activation_mean_shift: Optional[float] = None
    activation_var_ratio: Optional[float] = None


class RepresentationAlignmentEvaluator:
    """
    Evaluates representation alignment for MoE layers.
    
    Key metrics:
    1. Layer-level CKA: Compare combined layer output before/after expansion
       - Input: hidden_states to the layer
       - Pass through the full MoE layer (router + weighted expert combination)
       - Compare outputs
    
    2. Expert-to-Dense CKA: Compare each expert to original dense MLP
       - Requires storing the original dense MLP weights
       - Measures how well each expert preserves the original representation
    
    3. Inter-Expert Diversity: Pairwise CKA between all experts
       - Low diversity (high CKA) = experts are similar (might be redundant)
       - High diversity (low CKA) = experts are specialized
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def _compute_linear_cka(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        eps: float = 1e-12,
        verbose: bool = False
    ) -> float:
        """
        Compute Linear CKA between two activation matrices.
        
        CKA (Centered Kernel Alignment) measures similarity between representations.
        
        Args:
            X: Activations (samples × features)
            Y: Activations (samples × features)
            eps: Small value for numerical stability
            verbose: Print debug information
            
        Returns:
            CKA similarity score in [0, 1], or -1.0 on error
        """
        # Move to same device and convert to float32 for numerical stability
        X = X.detach().float()
        Y = Y.detach().float()
        
        # Ensure both are on the same device (prefer GPU if available)
        if X.device != Y.device:
            Y = Y.to(X.device)
        
        # Check for empty inputs
        if X.numel() == 0 or Y.numel() == 0:
            if verbose:
                print(f"         [CKA Error] Empty tensor: X.numel={X.numel()}, Y.numel={Y.numel()}")
            return -1.0
        
        # Ensure same number of samples
        if X.shape[0] != Y.shape[0]:
            if verbose:
                print(f"         [CKA Error] Sample mismatch: X.shape={X.shape}, Y.shape={Y.shape}")
            return -1.0
        
        n_samples = X.shape[0]
        
        # Check for very small sample size
        if n_samples < 2:
            if verbose:
                print(f"         [CKA Error] Too few samples: {n_samples}")
            return -1.0
        
        # Check variance BEFORE centering to catch constant outputs
        X_pre_var = X.var().item()
        Y_pre_var = Y.var().item()
        
        if verbose:
            print(f"         [CKA Debug] Pre-center: X_var={X_pre_var:.6e}, Y_var={Y_pre_var:.6e}")
        
        # Standard CKA: center first, then compute
        # Column-wise centering (subtract mean of each feature)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Check for near-zero variance after centering
        # This checks if samples differ from each other (row variation)
        X_var = (X ** 2).sum()
        Y_var = (Y ** 2).sum()
        
        if verbose:
            print(f"         [CKA Debug] Post-center: X_var={X_var.item():.6e}, Y_var={Y_var.item():.6e}")
        
        if X_var < eps or Y_var < eps:
            if verbose:
                # Provide more diagnostic info
                X_sample_std = X.std(dim=0).mean().item()  # Variation across samples
                Y_sample_std = Y.std(dim=0).mean().item()
                print(f"         [CKA Error] Near-zero variance after centering")
                print(f"         [CKA Diag] X mean_sample_std={X_sample_std:.6e}, Y mean_sample_std={Y_sample_std:.6e}")
                print(f"         [CKA Hint] This means all samples produce nearly identical outputs per feature")
            return -1.0
        
        # Use double precision for the critical computation
        X = X.double()
        Y = Y.double()
        
        # Compute using HSIC formulation for numerical stability
        # HSIC(X, Y) = trace(K_X @ H @ K_Y @ H) / (n-1)^2
        # where K_X = X @ X.T, H = I - 1/n * 11^T
        # 
        # Linear CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
        
        # For linear kernel: K = XX^T
        # Efficient computation: ||Y^T X||_F^2 / sqrt(||X^T X||_F * ||Y^T Y||_F)
        # But we need to be careful about centering
        
        # Gram matrices (kernel matrices)
        K_X = X @ X.T  # n x n
        K_Y = Y @ Y.T  # n x n
        
        # Center the Gram matrices
        # H @ K @ H where H = I - 1/n * 11^T
        one_n = torch.ones(n_samples, 1, device=X.device, dtype=X.dtype) / n_samples
        
        # Centered gram matrices: HKH = K - K@1/n - 1/n@K + 1/n@K@1/n
        K_X_centered = K_X - K_X.mean(dim=0, keepdim=True) - K_X.mean(dim=1, keepdim=True) + K_X.mean()
        K_Y_centered = K_Y - K_Y.mean(dim=0, keepdim=True) - K_Y.mean(dim=1, keepdim=True) + K_Y.mean()
        
        # HSIC values (unnormalized)
        hsic_xy = (K_X_centered * K_Y_centered).sum()
        hsic_xx = (K_X_centered * K_X_centered).sum()
        hsic_yy = (K_Y_centered * K_Y_centered).sum()
        
        if verbose:
            print(f"         [CKA Debug] HSIC_XY={hsic_xy.item():.6e}, HSIC_XX={hsic_xx.item():.6e}, HSIC_YY={hsic_yy.item():.6e}")
        
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        
        if denominator < eps:
            if verbose:
                print(f"         [CKA Error] Near-zero denominator: {denominator.item():.2e}")
            return -1.0
        
        cka = (hsic_xy / denominator).item()
        
        # CKA should be in [0, 1] for valid inputs, but numerical issues can cause small deviations
        return float(np.clip(cka, 0.0, 1.0))
    
    def compute_layer_output_cka(
        self,
        moe_layer,
        hidden_states: torch.Tensor,
        original_layer_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute CKA between the MoE layer's current output and a reference output.
        
        This measures how stable the layer's representation is after expansion.
        
        Args:
            moe_layer: The MoE layer module
            hidden_states: Input to the layer (batch, seq, hidden)
            original_layer_output: Reference output from before expansion
            attention_mask: Optional attention mask
            
        Returns:
            CKA score
        """
        with torch.no_grad():
            # Get current layer output
            current_output = moe_layer(hidden_states, attention_mask=attention_mask)
            if isinstance(current_output, tuple):
                current_output = current_output[0]
            
            # Flatten for CKA: (batch * seq, hidden)
            current_flat = current_output.view(-1, current_output.size(-1))
            original_flat = original_layer_output.view(-1, original_layer_output.size(-1))
            
            return self._compute_linear_cka(current_flat, original_flat)
    
    def compute_expert_activations(
        self,
        expert: nn.Module,
        hidden_states: torch.Tensor,
        activation_source: str = 'up_proj'
    ) -> torch.Tensor:
        """
        Get expert's activations from a specific layer.
        
        Args:
            expert: Expert module (typically has gate_proj, up_proj, down_proj)
            hidden_states: Input (samples, hidden_dim)
            activation_source: Which activation to extract:
                - 'output': Full MLP output (down_proj output, hidden_dim)
                - 'up_proj': Up projection output (intermediate_dim, recommended)
                - 'gate_proj': Gate projection output (intermediate_dim)
                - 'gate_up': Concatenation of gate and up (2 * intermediate_dim)
            
        Returns:
            Expert activations
        """
        with torch.no_grad():
            hidden_states = hidden_states.to(expert.gate_proj.weight.device)
            
            if activation_source == 'output':
                # Full output: down_proj(act_fn(gate_proj(x)) * up_proj(x))
                if hasattr(expert, 'down_proj'):
                    gate_out = expert.act_fn(expert.gate_proj(hidden_states))
                    up_out = expert.up_proj(hidden_states)
                    return expert.down_proj(gate_out * up_out)
                else:
                    return expert(hidden_states)
            
            elif activation_source == 'up_proj':
                # Just the up projection (recommended - higher variance)
                return expert.up_proj(hidden_states)
            
            elif activation_source == 'gate_proj':
                # Just the gate projection (after activation)
                return expert.act_fn(expert.gate_proj(hidden_states))
            
            elif activation_source == 'gate_up':
                # Concatenation of gate and up
                gate_out = expert.act_fn(expert.gate_proj(hidden_states))
                up_out = expert.up_proj(hidden_states)
                return torch.cat([gate_out, up_out], dim=-1)
            
            else:
                raise ValueError(f"Unknown activation_source: {activation_source}")
    
    def compute_expert_to_dense_cka(
        self,
        experts: List[nn.Module],
        dense_mlp: nn.Module,
        hidden_states: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compare each expert's output to the original dense MLP.
        
        This measures how well each expert preserves the original representation.
        
        Args:
            experts: List of expert modules
            dense_mlp: Original dense MLP module
            hidden_states: Input (samples, hidden_dim)
            
        Returns:
            Dictionary mapping expert_idx -> CKA score
        """
        results = {}
        
        with torch.no_grad():
            # Get dense MLP output as reference
            dense_output = self.compute_expert_activations(dense_mlp, hidden_states)
            
            for idx, expert in enumerate(experts):
                expert_output = self.compute_expert_activations(expert, hidden_states)
                cka = self._compute_linear_cka(expert_output, dense_output)
                results[idx] = cka
        
        return results
    
    def compute_inter_expert_diversity(
        self,
        experts: List[nn.Module],
        hidden_states: torch.Tensor,
        max_experts: int = 16,
        activation_source: str = 'up_proj',
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Compute pairwise CKA matrix between experts.
        
        High diversity (low pairwise CKA) is generally desirable as it indicates
        experts have specialized for different functions.
        
        Args:
            experts: List of expert modules
            hidden_states: Input (samples, hidden_dim)
            max_experts: Maximum number of experts to compare (for efficiency)
            activation_source: Which expert activation to use ('up_proj' recommended)
            verbose: Print debug information
            
        Returns:
            Tuple of (CKA matrix, diversity score)
            Diversity score = 1 - mean(off-diagonal CKA)
        """
        n_experts = min(len(experts), max_experts)
        cka_matrix = np.ones((n_experts, n_experts))
        
        if n_experts < 2:
            return cka_matrix, 0.0  # No diversity with 0 or 1 expert
        
        # Pre-compute all expert outputs
        expert_outputs = []
        with torch.no_grad():
            for idx, expert in enumerate(experts[:n_experts]):
                try:
                    output = self.compute_expert_activations(
                        expert, hidden_states, activation_source=activation_source
                    )
                    expert_outputs.append(output)
                    
                    if verbose and idx == 0:
                        print(f"      [Debug] Expert 0 activation shape: {output.shape}")
                        print(f"      [Debug] Expert 0 stats: mean={output.mean():.6f}, "
                              f"std={output.std():.6f}, min={output.min():.6f}, max={output.max():.6f}")
                except Exception as e:
                    if verbose:
                        print(f"      [Debug] Expert {idx} activation extraction failed: {e}")
                    expert_outputs.append(None)
        
        # Check if outputs are identical (common at initialization)
        identical_count = 0
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                if expert_outputs[i] is not None and expert_outputs[j] is not None:
                    diff = (expert_outputs[i] - expert_outputs[j]).abs().max().item()
                    if diff < 1e-6:
                        identical_count += 1
        
        total_pairs = n_experts * (n_experts - 1) // 2
        
        if identical_count == total_pairs:
            # All experts produce identical outputs (initialization state)
            if verbose:
                print(f"      [Debug] All {n_experts} experts produce identical outputs (CKA=1.0)")
                print(f"      [Debug] This is expected at initialization - diversity=0.0")
            return np.ones((n_experts, n_experts)), 0.0  # All identical = no diversity
        
        # Compute pairwise CKA
        valid_cka_count = 0
        error_count = 0
        first_error_printed = False
        
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                # Skip if either expert failed to produce activations
                if expert_outputs[i] is None or expert_outputs[j] is None:
                    error_count += 1
                    cka_matrix[i, j] = 0.0
                    cka_matrix[j, i] = 0.0
                    continue
                
                # Check if outputs are identical first
                diff = (expert_outputs[i] - expert_outputs[j]).abs().max().item()
                if diff < 1e-6:
                    # Identical outputs = CKA of 1.0
                    cka_matrix[i, j] = 1.0
                    cka_matrix[j, i] = 1.0
                    valid_cka_count += 1
                    continue
                
                # Verbose for first pair to debug
                pair_verbose = verbose and not first_error_printed and (i == 0 and j == 1)
                cka = self._compute_linear_cka(
                    expert_outputs[i], expert_outputs[j], verbose=pair_verbose
                )
                
                if cka < 0:  # Error code
                    error_count += 1
                    if verbose and not first_error_printed:
                        print(f"      [Debug] First CKA error at pair ({i}, {j})")
                        first_error_printed = True
                    cka = 1.0  # If CKA fails, likely identical - assume high similarity
                else:
                    valid_cka_count += 1
                    
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka
        
        if verbose:
            print(f"      [Debug] CKA computation: {valid_cka_count}/{total_pairs} valid, {error_count} errors")
        
        # Diversity score: 1 - mean of off-diagonal elements
        mask = ~np.eye(n_experts, dtype=bool)
        off_diag_values = cka_matrix[mask]
        
        mean_off_diag = off_diag_values.mean()
        diversity_score = 1.0 - mean_off_diag
        
        if verbose and error_count > 0:
            print(f"      [Debug] {error_count} CKA errors treated as identical (CKA=1.0)")
        
        return cka_matrix, diversity_score
    
    def compute_activation_stability(
        self,
        current_activations: torch.Tensor,
        reference_activations: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Measure activation stability between current and reference.
        
        Args:
            current_activations: Current layer activations
            reference_activations: Reference (before expansion) activations
            
        Returns:
            Tuple of (mean_shift, variance_ratio)
            - mean_shift: ||mean(current) - mean(reference)||
            - variance_ratio: var(current) / var(reference)
        """
        with torch.no_grad():
            current_flat = current_activations.view(-1, current_activations.size(-1))
            reference_flat = reference_activations.view(-1, reference_activations.size(-1))
            
            # Mean shift
            current_mean = current_flat.mean(dim=0)
            reference_mean = reference_flat.mean(dim=0)
            mean_shift = torch.norm(current_mean - reference_mean).item()
            
            # Variance ratio
            current_var = current_flat.var()
            reference_var = reference_flat.var()
            var_ratio = (current_var / (reference_var + 1e-10)).item()
            
            return mean_shift, var_ratio
    
    def evaluate_layer(
        self,
        moe_layer,
        hidden_states: torch.Tensor,
        original_dense_mlp: Optional[nn.Module] = None,
        reference_output: Optional[torch.Tensor] = None
    ) -> RepresentationMetrics:
        """
        Comprehensive evaluation of a single MoE layer.
        
        Args:
            moe_layer: The MoE layer (with .mlp containing router and experts)
            hidden_states: Input to the layer
            original_dense_mlp: Original dense MLP (for expert-to-dense CKA)
            reference_output: Reference layer output (for layer CKA and stability)
            
        Returns:
            RepresentationMetrics with all computed metrics
        """
        metrics = RepresentationMetrics()
        
        # Get experts
        mlp = moe_layer.mlp if hasattr(moe_layer, 'mlp') else moe_layer
        experts = getattr(mlp, 'scientific_experts', [])
        
        if len(experts) == 0:
            return metrics
        
        # Flatten hidden states for expert evaluation
        hidden_flat = hidden_states.view(-1, hidden_states.size(-1))
        
        # 1. Inter-expert diversity
        cka_matrix, diversity = self.compute_inter_expert_diversity(experts, hidden_flat)
        metrics.inter_expert_cka_matrix = cka_matrix
        metrics.expert_diversity_score = diversity
        
        # 2. Expert-to-dense CKA (if original MLP available)
        if original_dense_mlp is not None:
            metrics.expert_to_dense_cka = self.compute_expert_to_dense_cka(
                experts, original_dense_mlp, hidden_flat
            )
        
        # 3. Layer-level CKA (if reference output available)
        if reference_output is not None:
            with torch.no_grad():
                current_output = moe_layer(hidden_states)
                if isinstance(current_output, tuple):
                    current_output = current_output[0]
                
                current_flat = current_output.view(-1, current_output.size(-1))
                reference_flat = reference_output.view(-1, reference_output.size(-1))
                
                metrics.layer_cka = self._compute_linear_cka(current_flat, reference_flat)
                
                # Activation stability
                mean_shift, var_ratio = self.compute_activation_stability(
                    current_output, reference_output
                )
                metrics.activation_mean_shift = mean_shift
                metrics.activation_var_ratio = var_ratio
        
        return metrics


def print_representation_metrics(metrics: RepresentationMetrics, layer_idx: int):
    """Pretty print representation metrics."""
    print(f"\n📐 Representation Metrics for Layer {layer_idx}:")
    
    if metrics.expert_diversity_score is not None:
        print(f"   Expert Diversity Score: {metrics.expert_diversity_score:.4f}")
        print(f"   (1.0 = max diversity, 0.0 = all experts identical)")
    
    if metrics.layer_cka is not None:
        print(f"   Layer Output CKA (vs reference): {metrics.layer_cka:.4f}")
    
    if metrics.activation_mean_shift is not None:
        print(f"   Activation Mean Shift: {metrics.activation_mean_shift:.4f}")
        print(f"   Activation Variance Ratio: {metrics.activation_var_ratio:.4f}")
    
    if metrics.expert_to_dense_cka is not None:
        print(f"   Expert-to-Dense CKA:")
        for expert_idx, cka in metrics.expert_to_dense_cka.items():
            print(f"      Expert {expert_idx}: {cka:.4f}")
    
    if metrics.inter_expert_cka_matrix is not None:
        n = len(metrics.inter_expert_cka_matrix)
        print(f"   Inter-Expert CKA Matrix ({n}x{n}):")
        # Print summary statistics
        mask = ~np.eye(n, dtype=bool)
        off_diag = metrics.inter_expert_cka_matrix[mask]
        print(f"      Mean: {off_diag.mean():.4f}, Min: {off_diag.min():.4f}, Max: {off_diag.max():.4f}")

