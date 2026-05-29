#!/usr/bin/env python3
"""Visualize inference results from CL experiments.

Usage:
    python scripts/visualize_inference_results.py --results-dir <path_to_inference_results>
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# Task ordering
DATASETS = ["MeetingBank", "Py150", "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "C-STANCE"]
NUM_TASKS = 6


def load_result(results_dir: Path, round_idx: int, task_idx: int, task_name: str):
    """Load a single result file."""
    result_path = results_dir / f"results-{round_idx}-{task_idx}-{task_name}.json"
    if not result_path.exists():
        return None
    with open(result_path, 'r') as f:
        return json.load(f)


def get_metric_value(eval_dict, task_name):
    """Extract the main metric value from evaluation result."""
    if not eval_dict:
        return None

    if task_name == "MeetingBank":
        val = eval_dict.get("rouge-L", eval_dict.get("rougeL"))
        if val is not None:
            return val * 100 if val < 1 else val
    elif task_name == "Py150":
        val = eval_dict.get("similarity", eval_dict.get("EM"))
        return val
    elif task_name in ["NumGLUE-cm", "NumGLUE-ds"]:
        val = eval_dict.get("EM", eval_dict.get("accuracy"))
        if val is not None:
            return val * 100 if val < 1 else val
    elif task_name == "20Minuten":
        sari = eval_dict.get("sari", eval_dict.get("SARI"))
        if isinstance(sari, list) and len(sari) > 0:
            return sari[0].get("sari", sari[0]) if isinstance(sari[0], dict) else sari[0]
        return sari
    elif task_name == "C-STANCE":
        val = eval_dict.get("accuracy", eval_dict.get("F1", eval_dict.get("f1")))
        if val is not None:
            return val * 100 if val < 1 else val
    return None


def create_heatmap_data(results_dir: Path):
    """Create heatmap matrix from inference results."""
    # Matrix: rows = tasks, cols = rounds (after task X)
    # NaN for tasks not yet trained
    matrix = np.full((NUM_TASKS, NUM_TASKS), np.nan)

    for round_idx in range(NUM_TASKS):
        for task_idx in range(round_idx + 1):
            task_name = DATASETS[task_idx]
            result = load_result(results_dir, round_idx, task_idx, task_name)
            if result:
                eval_dict = result.get("eval", {})
                val = get_metric_value(eval_dict, task_name)
                if val is not None:
                    matrix[task_idx, round_idx] = val

    return matrix


def calculate_metrics(matrix):
    """Calculate Last Acc, Stage Acc, and BWT from the heatmap matrix.

    Paper formulas: A_b = (1/b)*sum_{j=1}^b A_{b,j}, Stage Acc = (1/N)*sum_{b=1}^N A_b.
    matrix[i,j] = performance on task i after round j (0-based);
    A_b = mean(matrix[0:b, b-1]), stage_acc = mean(A_1,...,A_N).

    Returns:
        avg: Last Accuracy = mean of last column
        stage_acc: Stage Accuracy = (1/N)*sum_b A_b per paper
        bwt: Backward Transfer = mean(last_col - diagonal) for first N-1 tasks
    """
    # Last Acc: mean of last column (final performance on all tasks)
    last_col = matrix[:, -1]
    avg = np.nanmean(last_col)

    # Stage Acc: \bar{A} = (1/N)*sum_{b=1}^N A_b,  A_b = (1/b)*sum_{j=1}^b A_{b,j}
    stage_acc_values = []
    for r in range(NUM_TASKS):
        Ab = np.nanmean(matrix[0 : (r + 1), r])
        if not np.isnan(Ab):
            stage_acc_values.append(Ab)
    stage_acc = np.mean(stage_acc_values) if stage_acc_values else np.nan

    # BWT: mean of (last_column[i] - diagonal[i]) for i in 0..N-2
    bwt_values = []
    for i in range(NUM_TASKS - 1):
        final_perf = matrix[i, -1]
        initial_perf = matrix[i, i]
        if not np.isnan(final_perf) and not np.isnan(initial_perf):
            bwt_values.append(final_perf - initial_perf)

    bwt = np.mean(bwt_values) if bwt_values else np.nan

    return avg, stage_acc, bwt


def create_custom_colormap():
    """Create colormap: 0=deep blue, 40=light gray/cream, 80=red."""
    colors = [
        (0.0, '#1565C0'),   # 0 - rich blue
        (0.25, '#64B5F6'),  # 20 - light blue
        (0.5, '#F5F5F5'),   # 40 - very light gray (almost white)
        (0.75, '#FF4444'), # 60 - red
        (1.0, '#C62828'),   # 80 - deeper red
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_blue_light_red',
        [(pos, color) for pos, color in colors]
    )
    return cmap


def plot_heatmap(matrix, title, output_path, cmap, vmin=0, vmax=80):
    """Plot a single heatmap and save to file."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask NaN values
    masked_matrix = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(range(NUM_TASKS))
    ax.set_xticklabels([f'After T{i}\n({DATASETS[i][:6]})' for i in range(NUM_TASKS)], fontsize=9)
    ax.set_yticks(range(NUM_TASKS))
    ax.set_yticklabels(DATASETS, fontsize=10)

    # Add value annotations
    for i in range(NUM_TASKS):
        for j in range(NUM_TASKS):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                # Choose text color based on background
                text_color = 'white' if val < 30 or val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=10, color=text_color, fontweight='bold')
            else:
                # Gray out untrained cells
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                            fill=True, facecolor='#E0E0E0', edgecolor='white'))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Training Stage', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (%)', fontsize=11)
    cbar.set_ticks([0, 20, 40, 60, 80])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def print_metrics_summary(matrix, avg, stage_acc, bwt):
    """Print a summary of metrics."""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print(f"\n{'Task':<15} {'After Learning':>15} {'Final':>15} {'Change':>15}")
    print("-" * 70)

    for i in range(NUM_TASKS):
        task_name = DATASETS[i]
        initial = matrix[i, i]
        final = matrix[i, -1]
        change = final - initial if not np.isnan(final) and not np.isnan(initial) else np.nan

        initial_str = f"{initial:.2f}" if not np.isnan(initial) else "N/A"
        final_str = f"{final:.2f}" if not np.isnan(final) else "N/A"
        change_str = f"{change:+.2f}" if not np.isnan(change) else "N/A"

        print(f"{task_name:<15} {initial_str:>15} {final_str:>15} {change_str:>15}")

    print("-" * 70)
    print(f"\n{'Last Acc (Avg):':<30} {avg:.2f}")
    print(f"{'Stage Acc (A_bar):':<30} {stage_acc:.2f}")
    print(f"{'Backward Transfer (BWT):':<30} {bwt:+.2f}")

    print("\n" + "=" * 70)
    print("Metrics Explanation:")
    print("  - Last Acc: (1/N)*sum_j A_{N,j} = mean of last column")
    print("  - Stage Acc: A_b=(1/b)*sum_{j=1}^b A_{b,j}, bar_A=(1/N)*sum_b A_b (per paper)")
    print("  - BWT: Mean of (final - initial) for first 5 tasks")
    print("=" * 70)


def plot_task_performance_trend(matrix, output_dir: Path):
    """Plot performance trend for each task across training rounds."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(range(NUM_TASKS))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for task_idx in range(NUM_TASKS):
        task_name = DATASETS[task_idx]
        # Get performance values for this task across rounds (starting from when it's trained)
        rounds = []
        values = []
        for round_idx in range(task_idx, NUM_TASKS):
            val = matrix[task_idx, round_idx]
            if not np.isnan(val):
                rounds.append(round_idx)
                values.append(val)

        if values:
            ax.plot(rounds, values, marker=markers[task_idx],
                   label=task_name, color=colors[task_idx],
                   linewidth=2, markersize=8)

    ax.set_xlabel('Training Round (After Task N)', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Task Performance Across Training Rounds\n(Tracking forgetting)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(NUM_TASKS))
    ax.set_xticklabels([f'T{i}' for i in range(NUM_TASKS)])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    output_path = output_dir / 'task_performance_trend.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_forgetting_bars(matrix, output_dir: Path):
    """Plot forgetting as bar chart for each task."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = []
    forgetting = []
    colors = []

    for i in range(NUM_TASKS - 1):  # Last task has no forgetting
        task_name = DATASETS[i]
        initial = matrix[i, i]
        final = matrix[i, -1]

        if not np.isnan(initial) and not np.isnan(final):
            change = final - initial
            tasks.append(task_name)
            forgetting.append(change)
            colors.append('#4CAF50' if change >= 0 else '#F44336')

    x = range(len(tasks))
    bars = ax.bar(x, forgetting, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, forgetting):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{val:+.1f}', ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11, rotation=15, ha='right')
    ax.set_ylabel('Performance Change (%)', fontsize=12)
    ax.set_title('Forgetting Analysis: Final - Initial Performance\n(Green = improvement, Red = forgetting)',
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add average BWT line
    avg_bwt = np.mean(forgetting)
    ax.axhline(y=avg_bwt, color='blue', linestyle='--', linewidth=2, label=f'Avg BWT: {avg_bwt:+.2f}')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'forgetting_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize inference results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to inference_results directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Name for the experiment (used in title)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Create output directory
    output_dir = results_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    # Determine experiment name
    exp_name = args.name if args.name else results_dir.parent.name

    print(f"Processing: {results_dir}")
    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print()

    # Create heatmap data
    print("Loading inference results...")
    matrix = create_heatmap_data(results_dir)

    # Calculate metrics (Last Acc, Stage Acc, BWT)
    avg, stage_acc, bwt = calculate_metrics(matrix)

    # Print summary
    print_metrics_summary(matrix, avg, stage_acc, bwt)

    # Create colormap
    cmap = create_custom_colormap()

    # Plot heatmap
    print("\nGenerating visualizations...")
    title = f"{exp_name}\nLast: {avg:.2f}  Stage: {stage_acc:.2f}  BWT: {bwt:+.2f}"
    plot_heatmap(matrix, title, output_dir / 'performance_heatmap.png', cmap)

    # Plot task performance trend
    plot_task_performance_trend(matrix, output_dir)

    # Plot forgetting bars
    plot_forgetting_bars(matrix, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
