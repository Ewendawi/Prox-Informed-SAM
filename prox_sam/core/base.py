import torch
import torch.nn as nn
from prox_sam.core.buffer import Buffer # Import buffer type if needed for type hinting or just use object
from prox_sam.core.utils import AverageMeter # Fix import

class BaseStrategy:
    def __init__(self, model, optimizer_cls, buffer, device, args, plugins=None):
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.buffer = buffer
        self.device = device
        self.args = args
        self.plugins = plugins or []
        
        self.task_id = 0
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Step counters for per-batch tracking
        self.global_step = 0
        self.step_in_task = 0
        
    def train_epoch(self, train_loader, **kwargs):
        raise NotImplementedError
        
    def train_task(self, train_loader, val_loaders, epochs):
        # 1. Optimizer init with Momentum and Weight Decay
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), 
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        # 2. Cosine Annealing Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.val_loaders = val_loaders or []
        
        # Reset step counter for this task
        self.step_in_task = 0

        
        # Hook: Before Task
        for p in self.plugins: p.before_train_task(self)
        
        for epoch in range(epochs):
            # Hook: Before Epoch (Not implemented in base for brevity)
            
            metrics = self.train_epoch(train_loader, epoch=epoch, total_epochs=epochs)
            
            # Step the scheduler at the end of each epoch
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
                
            losses = {}
            # Handle metrics for plugins
            if isinstance(metrics, dict):
                loss = metrics.get('loss', 0.0)
                self.current_metrics = metrics # For plugins
                for k, v in metrics.items():
                    if 'loss' in k:
                        losses[k] = v
            else:
                loss = metrics
                self.current_metrics = {'loss': loss}
                losses['loss'] = loss

            loss_str = ', '.join(f"{k}: {v:.4f}" for k, v in losses.items())
            
            print(f"Task {self.task_id}, Epoch {epoch}, {loss_str}")
            
            # Hook: After Epoch
            for p in self.plugins: p.after_train_epoch(self)
            
        # Hook: After Task
        for p in self.plugins: p.after_train_task(self)
        
        self.task_id += 1
