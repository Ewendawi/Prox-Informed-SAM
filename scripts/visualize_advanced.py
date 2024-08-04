import json
import os
import glob
import numpy as np
import math
import re
import csv
import matplotlib.pyplot as plt

from prox_sam.core.plugins import StabilityMetric

def mean_task_max_peak_drop(res):
    per_task = res.get('per_task_metrics', {})
    if not isinstance(per_task, dict) or not per_task:
        return None

    max_per_task = []
    for task_metrics in per_task.values():
        if not isinstance(task_metrics, dict):
            continue
        peak_drops = task_metrics.get('peak_drop', [])
        if isinstance(peak_drops, list) and peak_drops:
            max_per_task.append(max(peak_drops))

    if not max_per_task:
        return None
    return float(np.mean(max_per_task))

def recompute_and_update_metrics(
    results_path: str,
    metrics_path: str | None = None,
    recovery_window: int = 500,
    recovery_scale: float = 0.95,
    sg_window_steps: int = 100,
) -> dict:
    """
    Reads results.json, recomputes metrics, and updates metrics.json (merge, not replace).

    Args:
        results_path: Path to the results.json file
        metrics_path: Path to the metrics.json file. If None, derives from results_path
        recovery_window: Window for recovery time computation
        recovery_scale: Scale for recovery threshold (0.95 = 95% of ref accuracy)
        sg_window_steps: Window steps for stability gap area computation

    Returns:
        The updated metrics dict
    """
    results_path = os.path.abspath(results_path)

    # Derive metrics_path if not provided
    if metrics_path is None:
        metrics_path = results_path.replace("_results.json", "_metrics.json")

    # Load results.json
    print(f"Reading results from: {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Extract required data for metric computation
    acc_matrix = results.get("acc_matrix")
    acc_history = results.get("acc_history", {})
    task_switch_steps = results.get("task_switch_steps", [])

    if acc_matrix is None:
        raise ValueError("results.json must contain 'acc_matrix'")

    if not task_switch_steps:
        print("Warning: 'task_switch_steps' not found in results.json. "
              "Stability metrics (peak_drop, recovery_time, stability_gap) may be incomplete.")

    # Determine AUC bounds based on accuracy scale (fraction vs percent).
    acc_bounds = (0.0, 1.0)
    try:
        acc_arr = np.asarray(acc_matrix, dtype=float)
        if acc_arr.size and float(np.nanmax(acc_arr)) > 1.5:
            acc_bounds = (0.0, 100.0)
    except Exception:
        pass

    # Recompute metrics using StabilityMetric
    print("Recomputing metrics...")
    recomputed_metrics = StabilityMetric.compute_all_metrics_offline(
        acc_matrix=acc_matrix,
        acc_history=acc_history,
        task_switch_steps=task_switch_steps,
        recovery_window=recovery_window,
        recovery_scale=recovery_scale,
        sg_window_steps=sg_window_steps,
        acc_bounds=acc_bounds,
    )

    # Load existing metrics.json if it exists
    existing_metrics = {}
    if os.path.exists(metrics_path):
        print(f"Loading existing metrics from: {metrics_path}")
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)

    # Merge: recomputed values take precedence, but preserve any extra fields
    # from existing_metrics that are not in recomputed_metrics
    merged_metrics = {**existing_metrics, **recomputed_metrics}

    # Write updated metrics_recomputed.json
    metrics_recomputed_path = metrics_path.replace("_metrics.json", "_metrics_recomputed.json")
    print(f"Writing updated metrics to: {metrics_recomputed_path}")
    os.makedirs(os.path.dirname(metrics_recomputed_path), exist_ok=True)
    with open(metrics_recomputed_path, "w") as f:
        json.dump(merged_metrics, f, indent=2)

    print(f"Metrics updated successfully!")
    print(f"  - Forgetting: {merged_metrics['forgetting']:.2f}%")
    print(f"  - Learning Success: {merged_metrics['learning_success']:.2f}%")
    print(f"  - Stability Gap: {merged_metrics['stability_gap']:.2f}")
    print(f"  - Peak Drop: {merged_metrics['peak_drop']:.2f}%")
    print(f"  - Acc AUC (avg acc): {merged_metrics['acc_auc_norm_mean']:.4f}")
    print(f"  - Acc AUC ratio: {merged_metrics['acc_auc_ratio_01_mean']:.4f}")
    print(f"  - Recovery Time Median: {merged_metrics['recovery_time_median']:.0f} steps")
    print(f"  - Recovery Rate: {merged_metrics['recovery_rate']*100:.1f}%")

    return merged_metrics

def recompute_metrics_in_dir(
    results_dir: str,
    recovery_window: int = 500,
    recovery_scale: float = 0.95,
    sg_window_steps: int = 100,
) -> list[dict]:
    """
    Batch recompute metrics for all result files in a directory.

    Args:
        results_dir: Directory containing *_results.json files
        pattern: Glob pattern to match result files
        recovery_window: Window for recovery time computation
        recovery_scale: Scale for recovery threshold
        sg_window_steps: Window steps for stability gap computation

    Returns:
        List of recomputed metrics dicts
    """
    results_dir = os.path.abspath(results_dir)
    result_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    metrics_files = glob.glob(os.path.join(results_dir, "*_metrics.json"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return []

    print(f"Found {len(result_files)} result files to process")

    all_metrics = []
    for results_path, metrics_path in zip(result_files, metrics_files):
        try:
            metrics = recompute_and_update_metrics(
                results_path=results_path,
                metrics_path=metrics_path,
                recovery_window=recovery_window,
                recovery_scale=recovery_scale,
                sg_window_steps=sg_window_steps,
            )
            all_metrics.append(metrics)
            print()  # Blank line between files
        except Exception as e:
            print(f"Error processing {results_path}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print(f"Processed {len(all_metrics)} files successfully")
    return all_metrics

def load_results(results_dir, label_regex, id_regex):
    data = []
    files = glob.glob(os.path.join(results_dir, "*_results.json"))
    
    print(f"Found {len(files)} result files in {results_dir}")

    inr_group_id = 1

    for i, fpath in enumerate(files):
        try:
            with open(fpath, 'r') as f:
                res = json.load(f)

            metrics_path = fpath.replace("_results.json", "_metrics.json")
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
            fname = os.path.basename(fpath)

            if "baseline" in fname:
                group_id = "0"
                label = "Baseline"
            else:
                if label_regex:
                    match = re.match(label_regex, fname)
                    if match:
                        label = match.group(1)
                    else:
                        raise ValueError(f"No label regex match for {fname}")
                else:
                    raise ValueError(f"No label regex provided for {fname}")

                if id_regex:
                    group_id = re.match(id_regex, fname).group(1)
                else:
                    group_id = str(inr_group_id)
                    inr_group_id += 1

            # Extract metrics
            # 1. Final Average Accuracy
            acc_matrix = np.array(res['acc_matrix'])
            # acc_matrix is (N_tasks, N_tasks) usually, lower triangular
            # The final row contains accuracies of all tasks after the last task
            final_accs = acc_matrix[-1] 
            # Filter out zeros if task hasn't been reached (though this is final result file)
            # Assuming full run for visualization
            avg_acc = np.mean(final_accs)
            
            # 2. Learning Success (Avg of diagonal)
            learning_success = np.mean(np.diag(acc_matrix))

            # 2. Forgetting
            forgetting = metrics.get('forgetting', 0.0)

            # 3. Stability Gap Metrics
            sg = metrics.get('stability_gap', 0.0)
            peak_drop = metrics.get('peak_drop', 0.0)
            first_drop = metrics.get('first_drop', 0.0)
            last_drop = metrics.get('last_drop', 0.0)
            recovery_time_median = metrics.get('recovery_time_median', 0.0)
            recovery_rate = metrics.get('recovery_rate', 0.0)
            acc_auc_norm_mean = float(metrics.get('acc_auc_norm_mean', 0.0))
            acc_auc_ratio_01_mean = float(metrics.get('acc_auc_ratio_01_mean', 0.0))
            
            # 4. Learning Curve (step-wise or task-wise?)
            # The 'acc_history' might be granular. 'acc_matrix' diagonals are task learning.
            # Let's use 'acc_matrix' diagonal for "Task Learning" or row-means for "Avg Acc Evolution"
            # Row `i` in acc_matrix is the state after training on task `i`.
            # We want the average accuracy on tasks 0..i seen so far.
            avg_acc_history = []
            for i in range(len(acc_matrix)):
                row = acc_matrix[i]
                # Only value tasks 0..i
                valid_accs = row[:i+1]
                if len(valid_accs) > 0:
                    avg_acc_history.append(np.mean(valid_accs))
                else:
                    avg_acc_history.append(0)
            
            # 5. Prox Loss vs CE Loss (if available)
            # Just take the mean of the history if available
            metric_hist = res.get('metric_history', {})
            # This is usually a dict "0": {...}, "1": {...}
            avg_prox = 0.0
            avg_ce = 0.0
            total_steps = 0
            
            # Iterate through tasks to aggregate loss stats
            for t_idx in metric_hist:
                task_metrics = metric_hist[t_idx]
                if 'prox_loss' in task_metrics and 'ce_loss' in task_metrics:
                    p_loss = np.mean(task_metrics['prox_loss'])
                    c_loss = np.mean(task_metrics['ce_loss'])
                    # Simple average over tasks for a summary scalar
                    avg_prox += p_loss
                    avg_ce += c_loss
                    total_steps += 1
            
            if total_steps > 0:
                avg_prox /= total_steps
                avg_ce /= total_steps

            data.append({
                'id': group_id,
                'label': label,
                'avg_acc': avg_acc,
                'learning_success': float(metrics.get('learning_success', learning_success)),
                'forgetting': forgetting,
                'stability_gap': sg,
                'peak_drop': peak_drop,
                'first_drop': first_drop,
                'last_drop': last_drop,
                'recovery_time_median': recovery_time_median,
                'recovery_rate': recovery_rate,
                'acc_auc_norm_mean': acc_auc_norm_mean,
                'acc_auc_ratio_01_mean': acc_auc_ratio_01_mean,
                'avg_acc_history': avg_acc_history,
                'acc_history': res.get('acc_history', {}),
                'acc_matrix': acc_matrix,  # Store acc_matrix for plotting
                'per_task_metrics': metrics.get('per_task_metrics', {}),
                'avg_prox': avg_prox,
                'avg_ce': avg_ce,
                'debug_records': res.get('debug_records', []),
                'task_switch_steps': res.get('task_switch_steps', []),
                'config': res.get('config', {}),
            })
            
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")
            
    # Sort by group ID
    try:
        _ = float(data[0]['id'])
        data.sort(key=lambda x: float(x['id']))
    except ValueError:
        data.sort(key=lambda x: x['id'])
    return data

def plot_tradeoff_matrix(data, output_path):
    # 1. SG vs Forgetting (The Money Plot)
    # 2. Acc vs Forgetting
    # 3. Acc vs SG
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, item in enumerate(data):
        # Plot 1: SG vs Forgetting
        # Proposal V2 expectations: Unified (5) should be bottom-left (Low SG, Low F)
        axes[0].scatter(item['forgetting'], item['stability_gap'], label=item['label'], s=100)
        
        # Plot 2: Forgetting vs Accuracy
        axes[1].scatter(item['forgetting'], item['avg_acc'], label=item['label'], s=100)
        
        # Plot 3: Stability Gap vs Accuracy
        axes[2].scatter(item['stability_gap'], item['avg_acc'], label=item['label'], s=100)

    # Styling
    axes[0].set_xlabel('Forgetting (%) (Lower is better)')
    axes[0].set_ylabel('Stability Gap (SG) (Lower is better)')
    axes[0].set_title('Stability vs Plasticity Trade-off')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].set_xlabel('Forgetting (%) (Lower is better)')
    axes[1].set_ylabel('Avg Accuracy (%) (Higher is better)')
    axes[1].set_title('Accuracy vs Forgetting')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].set_xlabel('Stability Gap (SG) (Lower is better)')
    axes[2].set_ylabel('Avg Accuracy (%) (Higher is better)')
    axes[2].set_title('Accuracy vs Stability')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Legend only on first or last
    axes[0].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Trade-off Matrix to {output_path}")
    plt.close()

def plot_learning_curves(data, output_path):
    plt.figure(figsize=(10, 6))
    
    for i, item in enumerate(data):
        hist = item['avg_acc_history']
        tasks = range(1, len(hist) + 1)
        
        plt.plot(tasks, hist, label=item['label'], marker='.', linewidth=2)

    plt.xlabel('Number of Tasks Learned')
    plt.ylabel('Average Accuracy on Seen Tasks (%)')
    plt.title('Continual Learning Dynamics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Learning Curves to {output_path}")
    plt.close()

def plot_step_vs_acc_per_task(data, output_path):
    # One column with multiple rows: one subplot per task. Each subplot overlays all runs.
    all_task_ids = set()
    for item in data:
        acc_history = item.get('acc_history', {})
        if isinstance(acc_history, dict):
            all_task_ids.update(acc_history.keys())

    if not all_task_ids:
        print("No acc_history found; skipping step-vs-acc plot.")
        return

    def task_sort_key(task_id):
        try:
            return int(task_id)
        except Exception:
            return str(task_id)

    task_ids = sorted(all_task_ids, key=task_sort_key)
    nrows = len(task_ids)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(3, 2.4 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, task_id in zip(axes, task_ids):
        plotted_any = False
        for item in data:
            acc_history = item.get('acc_history', {})
            series = acc_history.get(task_id) if isinstance(acc_history, dict) else None
            if not isinstance(series, list) or not series:
                continue

            steps = []
            accs = []
            for pair in series:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                step, acc = pair
                try:
                    steps.append(float(step))
                    accs.append(float(acc))
                except Exception:
                    continue

            if steps:
                ax.plot(steps, accs, label=item['label'], linewidth=1.8, alpha=0.9)
                plotted_any = True

        ax.set_title(f"Task {task_id}: Step vs Accuracy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if plotted_any:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Step-vs-Acc per Task plot to {output_path}")
    plt.close()

def plot_acc_vs_task_from_matrix(data, output_path):
    """
    Plot accuracy vs task index (n) from acc_matrix.
    For each method/acc_matrix, create a subplot where:
    - X-axis: Task index (which task we just finished training)
    - Y-axis: Accuracy (%)
    - Each line represents one task's accuracy over time
    """
    n_methods = len(data)
    if n_methods == 0:
        print("No data to plot")
        return

    # Calculate max number of tasks across all runs
    max_tasks = max(item.get('acc_matrix', np.array([[]])).shape[1] for item in data if item.get('acc_matrix') is not None)

    # Calculate rows and columns for 2-column layout
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
    axes = axes.flatten() if n_methods > 1 else [axes]

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    for idx, item in enumerate(data):
        ax = axes[idx]
        acc_matrix = item.get('acc_matrix')
        if acc_matrix is None:
            continue

        n_tasks = acc_matrix.shape[1]

        # Plot each task's accuracy over time (each column of acc_matrix)
        for task_idx in range(n_tasks):
            # acc_matrix[task_idx:, task_idx] gives the accuracy of task `task_idx`
            # after it was first learned (skip upper triangular zeros/NaN)
            task_accs = acc_matrix[task_idx:, task_idx]
            # Filter out zeros/NaN values that indicate "not yet learned"
            valid_mask = (task_accs > 0) & (~np.isnan(task_accs))
            x_vals = np.arange(task_idx, task_idx + len(task_accs))[valid_mask]
            y_vals = task_accs[valid_mask]

            if len(x_vals) > 0:
                ax.plot(x_vals, y_vals, marker='o', label=f'Task {task_idx}', linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{item["label"]} - Task Accuracy Over Time')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, max_tasks - 0.5)

    # Set x-label on bottom row subplots
    for idx in range(len(axes)):
        if not axes[idx].get_visible():
            continue
        row_idx = idx // n_cols
        if row_idx == n_rows - 1:  # Bottom row
            axes[idx].set_xlabel('Task Index (n)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved Acc vs Task plot to {output_path}")
    plt.close()

def plot_task_metric_series_by_task(data, output_dir):
    """
    Plot per-task metric series with one figure per metric and subplots per task.
    Peak Drop and Stability Gap use raw per-task lists (x = switch-event index).
    Accuracy uses per-task accuracy over task index (from acc_matrix columns).
    """
    if not data:
        print("No data to plot")
        return

    def _normalize_task_id(task_id):
        try:
            return int(task_id)
        except Exception:
            return task_id

    def _task_sort_key(task_id):
        try:
            return (0, int(task_id))
        except Exception:
            return (1, str(task_id))

    def _get_task_ids(items):
        all_task_ids = set()
        for item in items:
            acc_matrix = item.get('acc_matrix')
            if isinstance(acc_matrix, np.ndarray) and acc_matrix.size:
                all_task_ids.update(range(acc_matrix.shape[1]))
            elif isinstance(acc_matrix, list) and len(acc_matrix) > 0:
                try:
                    all_task_ids.update(range(len(acc_matrix[0])))
                except Exception:
                    pass
            per_task = item.get('per_task_metrics', {})
            if isinstance(per_task, dict):
                all_task_ids.update(_normalize_task_id(k) for k in per_task.keys())
        return sorted(all_task_ids, key=_task_sort_key)

    task_ids = _get_task_ids(data)
    if not task_ids:
        print("No per-task data found; skipping task-metrics plot.")
        return

    def _prep_axes(nrows, title):
        fig, axes = plt.subplots(nrows, 1, figsize=(10, max(3, 2.4 * nrows)), sharex=False)
        if nrows == 1:
            axes = [axes]
        fig.suptitle(title)
        return fig, axes

    # Accuracy figure (per-task accuracy over task index)
    fig_acc, axes_acc = _prep_axes(len(task_ids), "Accuracy by Task (per config)")
    for ax, task_id in zip(axes_acc, task_ids):
        plotted_any = False
        for item in data:
            acc_matrix = item.get('acc_matrix')
            if acc_matrix is None:
                continue
            if isinstance(acc_matrix, list):
                acc_matrix = np.array(acc_matrix)
            if not isinstance(acc_matrix, np.ndarray) or not acc_matrix.size:
                continue
            if task_id >= acc_matrix.shape[1]:
                continue
            task_accs = acc_matrix[task_id:, task_id]
            valid_mask = (task_accs > 0) & (~np.isnan(task_accs))
            x_vals = np.arange(task_id, task_id + len(task_accs))[valid_mask]
            y_vals = task_accs[valid_mask]
            if len(x_vals) == 0:
                continue
            ax.plot(x_vals, y_vals, linewidth=1.8, label=item.get('label', 'config'))
            plotted_any = True
        ax.set_title(f"Task {task_id}")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if plotted_any:
            ax.legend(fontsize=8)
    axes_acc[-1].set_xlabel("Task Index")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "task_acc_by_task.png")
    plt.savefig(acc_path, dpi=150)
    print(f"Saved Task Accuracy by Task plot to {acc_path}")
    plt.close(fig_acc)

    def _plot_raw_metric(metric_key, title, y_label, filename):
        fig, axes = _prep_axes(len(task_ids), title)
        for ax, task_id in zip(axes, task_ids):
            plotted_any = False
            for item in data:
                per_task = item.get('per_task_metrics', {})
                per_task_norm = {}
                if isinstance(per_task, dict):
                    for k, v in per_task.items():
                        per_task_norm[_normalize_task_id(k)] = v
                task_metrics = per_task_norm.get(task_id, {})
                series = task_metrics.get(metric_key, [])
                if not isinstance(series, list) or not series:
                    continue
                x_vals = list(range(len(series)))
                y_vals = []
                for v in series:
                    try:
                        y_vals.append(float(v))
                    except Exception:
                        y_vals.append(float("nan"))
                ax.plot(x_vals, y_vals, marker='o', linewidth=1.6, label=item.get('label', 'config'))
                plotted_any = True
            ax.set_title(f"Task {task_id}")
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle='--', alpha=0.5)
            if plotted_any:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel("Switch-Event Index")
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved {title} plot to {out_path}")
        plt.close(fig)

    _plot_raw_metric(
        metric_key="peak_drop",
        title="Peak Drop by Task (per config)",
        y_label="Peak Drop (%)",
        filename="task_peak_drop_by_task.png",
    )
    _plot_raw_metric(
        metric_key="sg_area",
        title="Stability Gap by Task (per config)",
        y_label="Stability Gap (Area)",
        filename="task_sg_by_task.png",
    )

def plot_radar_summary(data, output_path):
    # Metrics to normalize:
    # 1. Accuracy (Higher is better)
    # 2. Forgetting (Lower is better)
    # 3. Stability Gap (Lower is better)

    # Extract raw values
    accs = [item['avg_acc'] for item in data]
    forgettings = [item['forgetting'] for item in data]
    sgs = [item['stability_gap'] for item in data]
    
    # Normalize to [0, 1] (Bigger is Better for all)
    # 1. Acc: (x - min) / (max - min)
    min_acc, max_acc = min(accs), max(accs)
    acc_scores = [(x - min_acc) / (max_acc - min_acc + 1e-9) for x in accs]
    
    # 2. Forgetting: 1 - (x - min)/(max - min)  => (max - x)/(max - min)
    min_f, max_f = min(forgettings), max(forgettings)
    f_scores = [(max_f - x) / (max_f - min_f + 1e-9) for x in forgettings]
    
    # 3. SG: 1 - (x - min)/(max - min) => (max - x)/(max - min)
    min_sg, max_sg = min(sgs), max(sgs)
    sg_scores = [(max_sg - x) / (max_sg - min_sg + 1e-9) for x in sgs]

    categories = ['Accuracy Score', 'Retention Score', 'Stability Score']
    N = len(categories)
    
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for i, item in enumerate(data):
        values = [acc_scores[i], f_scores[i], sg_scores[i]]
        
        # Close the loop
        values += values[:1]
        
        p = ax.plot(angles, values, linewidth=2, linestyle='solid', label=item['label'])
        color = p[0].get_color()
        ax.fill(angles, values, color=color, alpha=0.1)

    plt.xticks(angles[:-1], categories)
    # Set radial limit to [0, 1]
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Relative Performance (Normalized [0,1])')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Radar Chart to {output_path}")
    plt.close()

def plot_metric_comparison(data, output_path, metrics=['lr']):
    """
    Plot step vs metric(s) for each label (method), showing all tasks within each method.

    Args:
        data: List of result dictionaries
        output_path: Path to save the plot
        metrics: List of metric field names from debug_records
            - Single metric: plots on single y-axis
            - Two metrics: dual y-axis plot (left=first, right=second)
    """
    if not data:
        print("No data found; skipping metric comparison plot.")
        return

    n_labels = len(data)

    # Calculate grid layout (prefer roughly square)
    n_cols = int(np.ceil(np.sqrt(n_labels)))
    n_rows = int(np.ceil(n_labels / n_cols))

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=False)
    axes_flat = axes.flatten() if n_labels > 1 else [axes]

    # Hide unused subplots
    for idx in range(n_labels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Determine mode
    is_dual_axis = len(metrics) == 2
    is_single_axis = len(metrics) == 1

    if not (is_single_axis or is_dual_axis):
        print(f"metrics list must have 1 or 2 elements, got {len(metrics)}. Skipping plot.")
        return

    # Plot each label (method)
    for idx, item in enumerate(data):
        ax1 = axes_flat[idx]
        debug_records = item.get('debug_records', [])

        if not debug_records:
            continue

        # Collect all unique task_ids for this method
        task_ids = set()
        for record in debug_records:
            if isinstance(record, dict):
                task_id = record.get('task_id')
                if task_id is not None:
                    task_ids.add(task_id)

        # Sort task IDs for consistent ordering
        task_ids = sorted(task_ids)

        # Create twin axis once for dual-axis mode
        ax2 = None
        if is_dual_axis:
            ax2 = ax1.twinx()

        # Plot each task on this label's subplot
        for task_id in task_ids:
            steps = []
            metric1_values = []
            metric2_values = [] if is_dual_axis else None

            metric1_name = metrics[0]
            metric2_name = metrics[1] if is_dual_axis else None

            for record in debug_records:
                if isinstance(record, dict) and record.get('task_id') == task_id:
                    step = record.get('global_step')
                    val1 = record.get(metric1_name)
                    val2 = record.get(metric2_name) if is_dual_axis else None

                    # Filter based on mode
                    if is_single_axis:
                        if step is not None and val1 is not None:
                            steps.append(step)
                            metric1_values.append(val1)
                    else:  # dual_axis
                        if step is not None and val1 is not None and val2 is not None:
                            steps.append(step)
                            metric1_values.append(val1)
                            metric2_values.append(val2)

            if not steps:
                continue

            if is_single_axis:
                # Single y-axis plot
                ax1.plot(steps, metric1_values, label=f'Task {task_id}',
                         linewidth=2, linestyle='-', alpha=0.8)
            else:  # dual_axis
                # Plot metric1 on left y-axis (solid line)
                line1, = ax1.plot(steps, metric1_values, label=f'Task {task_id}',
                                  linewidth=2, linestyle='-', alpha=0.8)

                # Plot metric2 on right y-axis (dashed line, same color)
                ax2.plot(steps, metric2_values, color=line1.get_color(),
                         linewidth=2, linestyle='--', alpha=0.6)

        # Set subplot title and labels
        ax1.set_title(f"{item['label']}")
        ax1.set_xlabel('Step')
        ax1.set_ylabel(metrics[0].replace('_', ' ').title(), fontsize=9)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)

        if is_dual_axis:
            # Set the twin axis label for second metric
            ax2.set_ylabel(metrics[1].replace('_', ' ').title(), fontsize=9)
            ax2.tick_params(axis='y', labelsize=8)

        # Add legend (show tasks)
        lines1, labels1 = ax1.get_legend_handles_labels()
        if lines1:
            ax1.legend(lines1, labels1, loc='best', fontsize=7)

        ax1.grid(True, linestyle='--', alpha=0.5)

    # Overall figure title
    if is_single_axis:
        fig.suptitle(f'{metrics[0].replace("_", " ").title()} Per Method', fontsize=14, y=0.995)
    else:  # dual_axis
        fig.suptitle(f'{metrics[0].replace("_", " ").title()} vs {metrics[1].replace("_", " ").title()} Per Method',
                     fontsize=14, y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved metric comparison plot to {output_path}")
    plt.close()

def _load_regex(label):
    if label == 'replay_ratio':
        results_dir = 'results/sweep_replay_ratio'
        output_dir = 'results/sweep_replay_ratio'
        label_regex = r"(^\w+_\d+\.\d+)_results.*"
        id_regex = r"^\w+_(\d+\.\d+)_*"
    elif label == 'replay_size':
        results_dir = 'results/sweep_replay_size'
        output_dir = 'results/sweep_replay_size'
        label_regex = r"(^\w+_\d+)_results.*"
        id_regex = r"^\w+_(\d+)_*"
    elif label == 'tr_beta':
        results_dir = 'results/sweep_kl_tr_beta'
        output_dir = 'results/sweep_kl_tr_beta'
        label_regex = r"^kl_tr_(beta\d+\.\d+)_.*"
        id_regex = r"^kl_tr_beta(\d+\.\d+)_.*"
    elif label == 'prox':
        results_dir = 'results/sweep_prox'
        output_dir = 'results/sweep_prox'
        label_regex = r"^\d+_\w+_(lambda\d+\.\d+)_.*"
        id_regex = r"^\d+_\w+_lambda(\d+\.\d+)_.*"
    elif label == 'prox_schedule':
        results_dir = 'results/sweep_prox_schedule'
        output_dir = 'results/sweep_prox_schedule'
        label_regex = r"prox(.*)_.*"
        id_regex = r"^prox(\d+\.\d+).*_.*"
    elif label == 'sam':
        results_dir = 'results/sweep_sam'
        output_dir = 'results/sweep_sam'
        label_regex = r"^\d+_\w+_(rho\d+\.\d+)_.*"
        id_regex = r"^\d+_\w+_rho(\d+\.\d+)_.*"
    elif label == 'sam_debug':
        results_dir = 'results/sweep_sam_debug'
        output_dir = 'results/sweep_sam_debug'
        label_regex = r"^diag_sam_rho0\.05_(.*)_.*"
        id_regex = None
    elif label == 'sam_phase_rho':
        results_dir = 'results/sweep_sam_phase_rho'
        output_dir = 'results/sweep_sam_phase_rho'
        label_regex = r"^diag_phase_rho_(\w+_rho\d+\.\d+)_.*"
        id_regex = r"^diag_phase_rho_\w+_rho(\d+\.\d+)_.*"
    elif label == 'sam_ratio':
        results_dir = 'results/sweep_sam_ratio'
        output_dir = 'results/sweep_sam_ratio'
        label_regex = r"^diag_replay_ratio(.*)_.*"
        id_regex = r"^diag_replay_ratio(\d+\.\d+).*_.*"
    elif label == 'comb':
        results_dir = 'results/sweep_comb'
        output_dir = 'results/sweep_comb'
        label_regex = r"^\d+_\w+_(lambda\d+\.\d+_rho\d+\.\d+)_results.*"
        id_regex = r"^\d+_\w+_lambda(\d+\.\d+)_*"
    elif label == 'uni':
        results_dir = 'results/sweep_uni'
        output_dir = 'results/sweep_uni'
        label_regex = r"^\d+_\w+_(lambda\d+\.\d+_alpha\d+\.\d+)_results.*"
        id_regex = r"^\d+_\w+_lambda(\d+\.\d+)_*"
    elif label == 'uni_ratio':
        results_dir = 'results/sweep_uni_ratio'
        output_dir = 'results/sweep_uni_ratio'
        label_regex = r"^unified_gradratio_(lambda\d+\.\d+_beta\d+\.\d+)_results.*"
        id_regex = r"^unified_gradratio_.*beta(\d+\.\d+)_*"
    elif label == 'uni_ema':
        results_dir = 'results/sweep_uni_ema'
        output_dir = 'results/sweep_uni_ema'
        label_regex = r"^unified_ema_(lambda\d+\.\d+_gamma\d+\.\d+)_results.*"
        id_regex = r"^unified_ema_.*gamma(\d+\.\d+)_*"
    else:
        raise ValueError(f"Unknown label: {label}")
    return results_dir, output_dir, label_regex, id_regex    

def main():

    label = 'uni_ratio'

    results_dir, output_dir, label_regex, id_regex = _load_regex(label)

    data = load_results(results_dir, label_regex, id_regex)
    
    if not data:
        print("No data found!")
        return

    # Generate plots
    plot_tradeoff_matrix(data, os.path.join(output_dir, 'tradeoff_matrix.png'))
    plot_learning_curves(data, os.path.join(output_dir, 'learning_curve.png'))
    plot_step_vs_acc_per_task(data, os.path.join(output_dir, 'step_vs_acc_per_task.png'))
    plot_acc_vs_task_from_matrix(data, os.path.join(output_dir, 'acc_vs_task.png'))
    plot_task_metric_series_by_task(data, output_dir)
    plot_radar_summary(data, os.path.join(output_dir, 'radar_summary.png'))
    plot_metric_comparison(data, os.path.join(output_dir, 'lr_vs_param_drift.png'), metrics=['param_drift', 'lr'])
    plot_metric_comparison(data, os.path.join(output_dir, 'lambda_prox_vs_param_drift.png'), metrics=['param_drift', 'lambda_prox'])
    plot_metric_comparison(data, os.path.join(output_dir, 'ratio_r_rho.png'), metrics=['r_t', 'rho'])
    plot_metric_comparison(data, os.path.join(output_dir, 'ratio_g_p.png'), metrics=['g_norm_prox', 'p_norm_prox'])
    plot_metric_comparison(data, os.path.join(output_dir, 'rho.png'), metrics=['rho'])
    
    # Print Summary Table again for good measure
    print("\n| ID | Label | Avg Acc | LS | Forgetting | SG | Peak Drop | First Drop | Last Drop | AUC ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for item in data:
        print(f"| {item['id']} | {item['label']} | {item['avg_acc']:.2f} | {item['learning_success']:.2f} | {item['forgetting']:.2f} | {item['stability_gap']:.2f} | {item['peak_drop']:.2f} | {item['first_drop']:.2f} | {item['last_drop']:.2f} | {item.get('acc_auc_ratio_01_mean', 0.0):.4f} |")

    # Save to CSV
    csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'id',
            'label',
            'avg_acc',
            'learning_success',
            'forgetting',
            'stability_gap',
            'peak_drop',
            'first_drop',
            'last_drop',
            'acc_auc_ratio_01_mean',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in data:
            # Create a dict with only the fields we want
            row = {k: item[k] for k in fieldnames}
            writer.writerow(row)
    
    print(f"\nSaved summary CSV to {csv_path}")

if __name__ == "__main__":
    main()
