"""
Continual Learning Evaluation Metrics.

This module implements standard metrics for evaluating continual learning models,
as defined in the GEM paper (Lopez-Paz & Ranzato, 2017).

Metrics:
- ACC (Average Accuracy): Mean accuracy across all tasks after training
- BWT (Backward Transfer): How much learning new tasks affects old tasks
- FWT (Forward Transfer): How well knowledge transfers to future tasks

Reference:
- Gradient Episodic Memory for Continual Learning (Lopez-Paz & Ranzato, 2017)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import os


class CLMetrics:
    """
    Computes and tracks continual learning evaluation metrics.
    
    The metrics are based on a performance matrix R where:
    - R[i,j] = performance on task j after training on task i
    - Diagonal R[i,i] = performance on task i immediately after training on it
    
    Metrics:
    - ACC = (1/T) * sum(R[T,j] for j in 1..T)
    - BWT = (1/T-1) * sum(R[T,j] - R[j,j] for j in 1..T-1)
    - FWT = (1/T-1) * sum(R[i-1,i] - b[i] for i in 2..T)
    
    where b[i] is baseline (random) performance on task i.
    """
    
    def __init__(self, num_tasks: int, task_names: List[str] = None):
        """
        Initialize CL metrics tracker.
        
        Args:
            num_tasks: Total number of tasks in the continual learning sequence
            task_names: Optional list of task names
        """
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        
        # Performance matrix: R[i,j] = accuracy on task j after training round i
        # Initialized to NaN to indicate unmeasured entries
        self.R = np.full((num_tasks, num_tasks), np.nan)
        
        # Baseline performance (random model)
        self.baseline = np.zeros(num_tasks)
        
        # Optional: Store additional metrics per checkpoint
        self.additional_metrics: Dict[int, Dict[str, Any]] = {}
    
    def set_performance(self, train_round: int, task_idx: int, accuracy: float):
        """
        Record performance on task_idx after training round train_round.
        
        Args:
            train_round: The training round (0 to T-1)
            task_idx: The task index being evaluated (0 to T-1)
            accuracy: Accuracy or performance metric
        """
        if 0 <= train_round < self.num_tasks and 0 <= task_idx < self.num_tasks:
            self.R[train_round, task_idx] = accuracy
    
    def set_baseline(self, task_idx: int, baseline_accuracy: float):
        """
        Set baseline (random) performance for a task.
        
        Args:
            task_idx: Task index
            baseline_accuracy: Random model performance
        """
        if 0 <= task_idx < self.num_tasks:
            self.baseline[task_idx] = baseline_accuracy
    
    def load_from_predictions(self, predictions_dir: str, 
                               metric_key: str = 'accuracy',
                               result_pattern: str = 'results-{round}-{task_id}-{task_name}.json'):
        """
        Load performance from prediction results.
        
        Args:
            predictions_dir: Directory containing prediction result files
            metric_key: Key in result JSON for accuracy metric
            result_pattern: Pattern for result files
        """
        for round_idx in range(self.num_tasks):
            for task_idx in range(round_idx + 1):  # Only tasks seen so far
                task_name = self.task_names[task_idx]
                
                # Try different file patterns
                possible_files = [
                    os.path.join(predictions_dir, f"results-{round_idx}-{task_idx}-{task_name}.json"),
                    os.path.join(predictions_dir, f"round_{round_idx}", f"task_{task_idx}.json"),
                ]
                
                for filepath in possible_files:
                    if os.path.exists(filepath):
                        try:
                            with open(filepath, 'r') as f:
                                result = json.load(f)
                            
                            # Handle different result formats
                            if isinstance(result, dict):
                                if metric_key in result:
                                    accuracy = result[metric_key]
                                elif 'metrics' in result and metric_key in result['metrics']:
                                    accuracy = result['metrics'][metric_key]
                                elif 'score' in result:
                                    accuracy = result['score']
                                else:
                                    continue
                            else:
                                continue
                            
                            self.R[round_idx, task_idx] = accuracy
                            break
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
    
    def compute_acc(self) -> float:
        """
        Compute Average Accuracy (ACC).
        
        ACC = (1/T) * sum(R[T-1,j] for j in 0..T-1)
        
        Returns:
            Average accuracy across all tasks after final training round
        """
        final_round = self.num_tasks - 1
        final_row = self.R[final_round, :self.num_tasks]
        
        # Only count measured entries
        valid = ~np.isnan(final_row)
        if not np.any(valid):
            return np.nan
        
        return np.mean(final_row[valid])
    
    def compute_bwt(self) -> float:
        """
        Compute Backward Transfer (BWT).
        
        BWT = (1/(T-1)) * sum(R[T-1,j] - R[j,j] for j in 0..T-2)
        
        Positive BWT: Learning new tasks helps old tasks (positive transfer)
        Negative BWT: Learning new tasks hurts old tasks (forgetting)
        
        Returns:
            Backward transfer score
        """
        if self.num_tasks < 2:
            return 0.0
        
        final_round = self.num_tasks - 1
        bwt_scores = []
        
        for j in range(self.num_tasks - 1):  # Exclude last task
            R_final_j = self.R[final_round, j]  # Final performance on task j
            R_j_j = self.R[j, j]  # Performance right after training task j
            
            if not np.isnan(R_final_j) and not np.isnan(R_j_j):
                bwt_scores.append(R_final_j - R_j_j)
        
        if len(bwt_scores) == 0:
            return np.nan
        
        return np.mean(bwt_scores)
    
    def compute_fwt(self) -> float:
        """
        Compute Forward Transfer (FWT).
        
        FWT = (1/(T-1)) * sum(R[i-1,i] - baseline[i] for i in 1..T-1)
        
        Measures zero-shot transfer: how well does training on previous tasks
        help with future tasks (before training on them)?
        
        Returns:
            Forward transfer score
        """
        if self.num_tasks < 2:
            return 0.0
        
        fwt_scores = []
        
        for i in range(1, self.num_tasks):  # Skip first task
            R_prev_i = self.R[i-1, i]  # Performance on task i before training on it
            baseline_i = self.baseline[i]
            
            if not np.isnan(R_prev_i):
                fwt_scores.append(R_prev_i - baseline_i)
        
        if len(fwt_scores) == 0:
            return np.nan
        
        return np.mean(fwt_scores)
    
    def compute_forgetting(self) -> float:
        """
        Compute average forgetting (alternative to BWT).
        
        Forgetting = (1/(T-1)) * sum(max_{l<=i}(R[l,j]) - R[T-1,j] for j in 0..T-2)
        
        Measures the gap between best historical performance and final performance.
        
        Returns:
            Average forgetting score (lower is better)
        """
        if self.num_tasks < 2:
            return 0.0
        
        final_round = self.num_tasks - 1
        forgetting_scores = []
        
        for j in range(self.num_tasks - 1):  # Exclude last task
            # Find max performance on task j across all rounds where j was seen
            max_perf = np.nan
            for l in range(j, self.num_tasks):
                if not np.isnan(self.R[l, j]):
                    if np.isnan(max_perf) or self.R[l, j] > max_perf:
                        max_perf = self.R[l, j]
            
            final_perf = self.R[final_round, j]
            
            if not np.isnan(max_perf) and not np.isnan(final_perf):
                forgetting_scores.append(max_perf - final_perf)
        
        if len(forgetting_scores) == 0:
            return np.nan
        
        return np.mean(forgetting_scores)
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all CL metrics.
        
        Returns:
            Dictionary with ACC, BWT, FWT, and Forgetting
        """
        return {
            'ACC': self.compute_acc(),
            'BWT': self.compute_bwt(),
            'FWT': self.compute_fwt(),
            'Forgetting': self.compute_forgetting()
        }
    
    def get_task_specific_bwt(self) -> Dict[str, float]:
        """
        Get BWT for each individual task.
        
        Returns:
            Dictionary mapping task names to their BWT scores
        """
        final_round = self.num_tasks - 1
        task_bwt = {}
        
        for j in range(self.num_tasks - 1):
            R_final_j = self.R[final_round, j]
            R_j_j = self.R[j, j]
            
            if not np.isnan(R_final_j) and not np.isnan(R_j_j):
                task_bwt[self.task_names[j]] = R_final_j - R_j_j
            else:
                task_bwt[self.task_names[j]] = np.nan
        
        return task_bwt
    
    def save(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Output filepath
        """
        metrics = self.compute_all()
        task_bwt = self.get_task_specific_bwt()
        
        output = {
            'num_tasks': self.num_tasks,
            'task_names': self.task_names,
            'metrics': metrics,
            'task_specific_bwt': task_bwt,
            'performance_matrix': self.R.tolist(),
            'baseline': self.baseline.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CLMetrics':
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Input filepath
            
        Returns:
            CLMetrics instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = cls(
            num_tasks=data['num_tasks'],
            task_names=data.get('task_names')
        )
        metrics.R = np.array(data['performance_matrix'])
        metrics.baseline = np.array(data.get('baseline', np.zeros(data['num_tasks'])))
        
        return metrics
    
    def __repr__(self) -> str:
        metrics = self.compute_all()
        return (f"CLMetrics(ACC={metrics['ACC']:.4f}, "
                f"BWT={metrics['BWT']:.4f}, "
                f"FWT={metrics['FWT']:.4f}, "
                f"Forgetting={metrics['Forgetting']:.4f})")


def evaluate_cl_run(predictions_dir: str, task_names: List[str], 
                    output_path: str = None) -> Dict[str, float]:
    """
    Convenience function to evaluate a complete CL run.
    
    Args:
        predictions_dir: Directory with prediction results
        task_names: List of task names
        output_path: Optional path to save results
        
    Returns:
        Dictionary with all CL metrics
    """
    metrics = CLMetrics(num_tasks=len(task_names), task_names=task_names)
    metrics.load_from_predictions(predictions_dir)
    
    results = metrics.compute_all()
    
    if output_path:
        metrics.save(output_path)
    
    return results


# =============================================================================
# CKA-specific metrics integration
# =============================================================================

class CKACLMetrics(CLMetrics):
    """
    Extended CL metrics with CKA alignment tracking.
    
    Tracks both accuracy-based metrics and CKA alignment scores.
    """
    
    def __init__(self, num_tasks: int, task_names: List[str] = None):
        super().__init__(num_tasks, task_names)
        
        # CKA matrix: CKA[i,j,l] = CKA score for task j at layer l after round i
        self.cka_scores: Dict[int, Dict[int, Dict[int, float]]] = {}
        # Structure: {round_idx: {task_idx: {layer_idx: cka_score}}}
    
    def set_cka_score(self, train_round: int, task_idx: int, layer_idx: int, cka: float):
        """Record CKA score for a specific configuration."""
        if train_round not in self.cka_scores:
            self.cka_scores[train_round] = {}
        if task_idx not in self.cka_scores[train_round]:
            self.cka_scores[train_round][task_idx] = {}
        self.cka_scores[train_round][task_idx][layer_idx] = cka
    
    def load_cka_from_metrics(self, metrics_dir: str):
        """
        Load CKA scores from expert metrics v2 evaluation.
        
        Args:
            metrics_dir: Directory with expert_metrics_v2 results
        """
        for round_idx in range(self.num_tasks):
            metrics_file = os.path.join(metrics_dir, f"checkpoint_{round_idx}_metrics.json")
            
            if not os.path.exists(metrics_file):
                continue
            
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                cka_data = data.get('cka', {})
                
                for layer_key, layer_data in cka_data.items():
                    layer_idx = int(layer_key.replace('layer_', ''))
                    
                    for task_key, task_data in layer_data.items():
                        # Parse task index from task_key
                        task_idx = self._parse_task_key(task_key)
                        if task_idx is not None:
                            cka_score = task_data.get('cka_score', np.nan)
                            self.set_cka_score(round_idx, task_idx, layer_idx, cka_score)
                            
            except (json.JSONDecodeError, KeyError):
                continue
    
    def _parse_task_key(self, task_key: str) -> Optional[int]:
        """Parse task index from task key string."""
        for i, name in enumerate(self.task_names):
            if name in task_key:
                return i
        return None
    
    def compute_cka_bwt(self, layer_idx: int = None) -> float:
        """
        Compute BWT based on CKA scores.
        
        CKA-BWT = (1/(T-1)) * sum(CKA[T-1,j] - CKA[j,j] for j in 0..T-2)
        
        Args:
            layer_idx: Specific layer to compute, or None for average across layers
            
        Returns:
            CKA-based backward transfer
        """
        if self.num_tasks < 2:
            return 0.0
        
        final_round = self.num_tasks - 1
        cka_bwt_scores = []
        
        for j in range(self.num_tasks - 1):
            if layer_idx is not None:
                # Specific layer
                cka_final = self.cka_scores.get(final_round, {}).get(j, {}).get(layer_idx)
                cka_initial = self.cka_scores.get(j, {}).get(j, {}).get(layer_idx)
                
                if cka_final is not None and cka_initial is not None:
                    cka_bwt_scores.append(cka_final - cka_initial)
            else:
                # Average across layers
                layer_scores = []
                for l in self._get_all_layers():
                    cka_final = self.cka_scores.get(final_round, {}).get(j, {}).get(l)
                    cka_initial = self.cka_scores.get(j, {}).get(j, {}).get(l)
                    
                    if cka_final is not None and cka_initial is not None:
                        layer_scores.append(cka_final - cka_initial)
                
                if layer_scores:
                    cka_bwt_scores.append(np.mean(layer_scores))
        
        if len(cka_bwt_scores) == 0:
            return np.nan
        
        return np.mean(cka_bwt_scores)
    
    def _get_all_layers(self) -> List[int]:
        """Get all unique layer indices in the CKA data."""
        layers = set()
        for round_data in self.cka_scores.values():
            for task_data in round_data.values():
                layers.update(task_data.keys())
        return sorted(layers)
    
    def compute_all_with_cka(self) -> Dict[str, float]:
        """
        Compute all metrics including CKA-based metrics.
        
        Returns:
            Dictionary with standard and CKA-based metrics
        """
        results = self.compute_all()
        results['CKA_BWT'] = self.compute_cka_bwt()
        
        # Per-layer CKA BWT
        for layer in self._get_all_layers():
            results[f'CKA_BWT_layer_{layer}'] = self.compute_cka_bwt(layer_idx=layer)
        
        return results
    
    def save(self, filepath: str):
        """Save metrics including CKA data."""
        metrics = self.compute_all_with_cka()
        task_bwt = self.get_task_specific_bwt()
        
        output = {
            'num_tasks': self.num_tasks,
            'task_names': self.task_names,
            'metrics': metrics,
            'task_specific_bwt': task_bwt,
            'performance_matrix': self.R.tolist(),
            'baseline': self.baseline.tolist(),
            'cka_scores': self._serialize_cka_scores()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
    
    def _serialize_cka_scores(self) -> Dict:
        """Convert CKA scores dict to JSON-serializable format."""
        result = {}
        for round_idx, round_data in self.cka_scores.items():
            result[str(round_idx)] = {}
            for task_idx, task_data in round_data.items():
                result[str(round_idx)][str(task_idx)] = {
                    str(k): v for k, v in task_data.items()
                }
        return result


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute CL metrics')
    parser.add_argument('--predictions-dir', type=str, required=True,
                        help='Directory with prediction results')
    parser.add_argument('--metrics-dir', type=str, default=None,
                        help='Directory with expert metrics v2 results (for CKA)')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated task names')
    parser.add_argument('--output', type=str, default='cl_metrics.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    task_names = args.tasks.split(',')
    
    if args.metrics_dir:
        # Use CKA-enhanced metrics
        metrics = CKACLMetrics(num_tasks=len(task_names), task_names=task_names)
        metrics.load_from_predictions(args.predictions_dir)
        metrics.load_cka_from_metrics(args.metrics_dir)
        results = metrics.compute_all_with_cka()
    else:
        # Standard metrics
        metrics = CLMetrics(num_tasks=len(task_names), task_names=task_names)
        metrics.load_from_predictions(args.predictions_dir)
        results = metrics.compute_all()
    
    metrics.save(args.output)
    
    print("\nContinual Learning Metrics:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
