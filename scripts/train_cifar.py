import argparse
import json
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from prox_sam.models.resnet import IncrementalResNet
from prox_sam.core.buffer import Buffer
from prox_sam.core.utils import set_seed
from prox_sam.methods.prox_sam import ProxSAM
from prox_sam.core.plugins import StabilityMetric
from prox_sam.datasets.cifar import get_split_cifar100

def get_parser():
    parser = argparse.ArgumentParser()
    # Exp Settings
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay L2 penalty')

    # Prox
    parser.add_argument('--lambda_prox', type=float, default=1.0, help='Proximal strength')
    parser.add_argument(
        '--lambda_schedule',
        type=str,
        default='constant',
        choices=['constant', 'inc_linear', 'inc_cosine', 'inc_poly'],
        help='Lambda schedule for prox',
    )
    parser.add_argument('--lambda_warmup_power', type=float, default=1.0, help='Power for inc schedule')
    parser.add_argument('--prox_start_epoch', type=int, default=0, help='Delay prox schedule start within each task')
    parser.add_argument('--prox_backbone_only', action='store_true', help='Apply proximal only to backbone')
    parser.add_argument('--prox_exclude_bn', action='store_true', help='Exclude BN affine params from prox')
    
    # SAM 
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--sam_epochs', type=int, default=5, help='How many epochs to use SAM')
    parser.add_argument('--sam_phase', type=str, default='last', choices=['first', 'last', 'all'], help='When to apply SAM within each task')
    parser.add_argument('--rho_base', type=float, default=0.05, help='Fixed rho if no binding')
    parser.add_argument('--sam_bn_mode', type=str, default='momentum0', choices=['momentum0', 'save_restore', 'none'], help='BN handling in SAM second pass')
    parser.add_argument('--sam_batch_mode', type=str, default='split', choices=['all', 'current', 'split'], help='Batch used for SAM (all=current+replay, current=current-only, split=SAM on current + normal grad on replay)')

    # Unified
    parser.add_argument('--use_coupling', action='store_true', help='Use Prox-Informed coupling')
    parser.add_argument('--coupling_mode', type=str, default='delta_ema', choices=['delta_ema', 'grad_ratio'], help='How to bind rho when use_coupling is enabled')
    parser.add_argument('--delta_gamma', type=float, default=None, help='Gamma in rho = gamma * EMA(delta)')
    parser.add_argument('--delta_ema_beta', type=float, default=0.1, help='EMA beta for delta (drift norm) used in rho binding')
    parser.add_argument('--grad_ratio_beta', type=float, default=1.0, help='Beta in rho = rho_max/(1+beta*r) for grad_ratio coupling')
    parser.add_argument('--grad_ratio_eps', type=float, default=1e-12, help='Epsilon for r = ||p||/(||g||+eps) in grad_ratio coupling')

    # Debug / Diagnostics
    parser.add_argument('--debug_prox', action='store_true', help='Record per-step prox diagnostics into results JSON')
    parser.add_argument('--debug_sam', action='store_true', help='Record per-step SAM diagnostics into results JSON')
    parser.add_argument('--debug_every_n_steps', type=int, default=1, help='Record diagnostics every N steps (only if debug_sam)')
    
    # Metrics / Logging
    parser.add_argument('--recovery_window', type=int, default=500, help='Recovery window in steps for stability metrics')
    parser.add_argument('--recovery_scale', type=float, default=0.95, help='Recovery threshold scale (fraction of pre-switch acc)')
    parser.add_argument('--eval_every_n_batches', type=int, default=50, help='Run eval every N batches (set >1 to reduce overhead)')

    # Replay
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--replay_ratio', type=float, default=0.5, help='Fraction of each step batch reserved for replay')
    
    return parser

@dataclass
class trainConfig:
    name: str = 'cifar'
    data_root: str = './data'
    output_dir: str = './results'
    seed: int = 42
    n_tasks: int = 10
    batch_size: int = 32
    epochs: int = 10

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    lambda_prox: float = 0.0
    lambda_schedule: str = 'constant'
    lambda_warmup_power: float = 1.0
    prox_start_epoch: int = 0
    prox_backbone_only: bool = True
    prox_exclude_bn: bool = False

    use_sam: bool = False
    sam_epochs: int = 5
    sam_backbone_only: bool = True
    rho_base: float = 0.05
    rho_min: float = 1e-4
    rho_max: float = 20.0
    sam_phase: str = 'last'
    sam_bn_mode: str = 'momentum0'
    sam_batch_mode: str = 'split'

    use_coupling: bool = False
    coupling_mode: str = 'delta_ema'
    delta_gamma: float | None = None
    delta_ema_beta: float = 0.1
    grad_ratio_beta: float = 1.0
    grad_ratio_eps: float = 1e-12

    debug_prox: bool = False
    debug_sam: bool = False
    debug_every_n_steps: int = 50

    recovery_window: int = 5000
    recovery_scale: float = 0.80
    eval_every_n_batches: int = 200

    buffer_size: int = 5000
    replay_ratio: float = 0.2

    train_per_task: int = 2000
    test_per_task: int = 400

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
    
    def __to_dict__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def train(config: trainConfig):
    set_seed(config.seed)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    device = torch.device(device)
    print(f"Running on {device} with seed {config.seed}")
    
    # Data
    print("Preparing Split CIFAR-100...")
    train_datasets, test_datasets = get_split_cifar100(config.n_tasks, config.seed, data_root=config.data_root, max_train_per_task=config.train_per_task, max_test_per_task=config.test_per_task)
    
    # Model & Buffer
    model = IncrementalResNet(n_initial_classes=100 // config.n_tasks)
    model.to(device)
    buffer = Buffer(config.buffer_size, (3, 32, 32), device=device)
    
    # Plugins (Metrics)
    metric_plugin = StabilityMetric(
        config.n_tasks,
        eval_every_n_batches=config.eval_every_n_batches,
    )
    
    # Strategy
    strategy = ProxSAM(model, optim.SGD, buffer, device, config, plugins=[metric_plugin])
    
    # --- Loop ---
    for t_id in range(config.n_tasks):
        print(f"\n=== Starting Task {t_id} ===")
        
        # 1. Expand Classifier logic
        if t_id > 0:
             n_new = 100 // config.n_tasks
             strategy.model.expand_classifier(n_new)
             strategy.model.to(device)
             
        # 2. Train
        # Keep per-step total batch budget fixed at config.batch_size by allocating a portion
        # of each step to replay, and using the remainder for current-task samples.
        replay_ratio = float(getattr(config, "replay_ratio", 0.5))
        replay_ratio = max(0.0, min(1.0, replay_ratio))
        current_batch_size = int(round(config.batch_size * (1.0 - replay_ratio)))
        current_batch_size = max(1, min(config.batch_size, current_batch_size))
        train_loader = DataLoader(
            train_datasets[t_id],
            batch_size=current_batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Prepare val loaders for online eval (all previous tasks + current task)
        val_loaders = []
        for k in range(t_id + 1):
             val_loaders.append(DataLoader(test_datasets[k], batch_size=100, shuffle=False))
             
        strategy.train_task(train_loader, val_loaders, config.epochs)
        
        # 3. Buffer Update
        # Dynamically cap per-task additions so total buffer budget stays comparable across tasks.
        # (Exact per-task balance is handled by the sampling policy; this just controls ingestion.)
        per_task_cap = max(1, int(config.buffer_size // config.n_tasks))
        samples_collected = 0
        for inputs, targets in train_loader:
             remaining = per_task_cap - samples_collected
             if remaining <= 0:
                 break
             if inputs.size(0) > remaining:
                 inputs = inputs[:remaining]
                 targets = targets[:remaining]
             buffer.add(inputs, targets, task_id=t_id)
             samples_collected += int(inputs.size(0))
        
        # 4. Metrics (retrieve online-eval accuracies from plugin)
        accs = {k: float(metric_plugin.acc_matrix[t_id, k]) for k in range(t_id + 1)}
        print(f"Task {t_id} Result: {accs}")
        
    print("\n=== Final Results ===")
    recovery_window = config.recovery_window
    recovery_scale = config.recovery_scale
    sg_window_steps = 100
    all_metrics = StabilityMetric.compute_all_metrics_offline(
        metric_plugin.acc_matrix,
        metric_plugin.acc_history,
        metric_plugin.task_switch_steps,
        recovery_window=recovery_window,
        recovery_scale=recovery_scale,
        sg_window_steps=sg_window_steps,
        acc_bounds=(0.0, 100.0),
    )
    print(f"Avg Acc (Final): {np.mean(metric_plugin.acc_matrix[-1]):.2f}")
    print(f"Learning Success (LS): {all_metrics['learning_success']:.2f}")
    print(f"Retention Ratio: {all_metrics['retention_ratio']:.2f}")
    print(f"Avg Forgetting: {all_metrics['forgetting']:.2f}")
    print(f"Stability Gap: {all_metrics['stability_gap']:.2f}")
    print(f"Peak Drop: {all_metrics['peak_drop']:.2f}")
    print(f"First Drop: {all_metrics['first_drop']:.2f}")
    print(f"Last Drop: {all_metrics['last_drop']:.2f}")
    print(f"Recovery Median: {all_metrics['recovery_time_median']:.2f} steps")
    print(f"Recovery Rate: {all_metrics['recovery_rate']*100:.1f}%")
    print(f"Acc AUC ratio: {all_metrics['acc_auc_ratio_01_mean']:.3f}")
    
    # Save Results
    results_dir = config.output_dir
    os.makedirs(results_dir, exist_ok=True)
    raw_results = metric_plugin.to_dict()
    raw_results["config"] = vars(config)
    if getattr(strategy, "debug_records", None):
        raw_results["debug_records"] = strategy.debug_records
    with open(f"{results_dir}/{config.name}_results.json", "w") as f:
        json.dump(raw_results, f, indent=4)

    metric_results = {
        "forgetting": all_metrics["forgetting"],
        "learning_success": all_metrics["learning_success"],
        "retention_ratio": all_metrics["retention_ratio"],
        "stability_gap": all_metrics["stability_gap"],
        "peak_drop": all_metrics["peak_drop"],
        "first_drop": all_metrics["first_drop"],
        "last_drop": all_metrics["last_drop"],
        "recovery_time_median": all_metrics["recovery_time_median"],
        "recovery_rate": all_metrics["recovery_rate"],
        "recovery_window": all_metrics["recovery_window"],
        "recovery_scale": all_metrics["recovery_scale"],
        "sg_window_steps": all_metrics["sg_window_steps"],
        "acc_bounds": all_metrics.get("acc_bounds"),
        "acc_auc_norm_mean": all_metrics["acc_auc_norm_mean"],
        "acc_auc_ratio_01_mean": all_metrics["acc_auc_ratio_01_mean"],
        "per_task_metrics": all_metrics["per_task_metrics"],
        "all_sg_areas": all_metrics["all_sg_areas"],
        "all_peak_drops": all_metrics["all_peak_drops"],
        "all_first_drops": all_metrics["all_first_drops"],
        "all_last_drops": all_metrics["all_last_drops"],
        "all_recovery_times": all_metrics["all_recovery_times"],
        "switch_steps": all_metrics["switch_steps"],
        "final_step": all_metrics["final_step"],
        "config": config.__to_dict__(),
    }
    with open(f"{results_dir}/{config.name}_metrics.json", "w") as f:
        json.dump(metric_results, f, indent=4)

    print(f"Raw data saved to: {results_dir}/{config.name}_results.json")
    print(f"Metric results saved to: {results_dir}/{config.name}_metrics.json")

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)

    config = trainConfig(args)

    def is_debug_mode():
        import os
        import sys
        return (
            sys.gettrace() is not None or
            'pydevd' in sys.modules or
            'VSCODE_PID' in os.environ
        )
    if is_debug_mode():
        config.n_tasks = 5
        config.epochs = 3
        config.seed = 4201
        config.output_dir = 'results/debug'

        config.lambda_prox = 0.1
        config.prox_backbone_only = True 
        config.lambda_schedule = "inc_linear"
        config.debug_prox = True
        config.prox_start_epoch = 0 
        config.prox_exclude_bn = True

        config.use_sam = True
        config.sam_epochs = 1
        config.rho_base = 0.5

        config.use_decoupled = True

        # config.coupling_mode = "delta_ema"
        # config.delta_ema_beta = 0.1
        # config.delta_gamma = 1.0

        config.coupling_mode = "grad_ratio"
        config.grad_ratio_alpha = 0.5
        config.grad_ratio_beta = 0.5

        config.train_per_task = 1000
        config.test_per_task = 200

    train(config)
