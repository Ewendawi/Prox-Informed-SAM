import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Buffer:
    """
    Simple Experience Replay Buffer.
    Stores samples and targets.
    """
    def __init__(self, buffer_size, input_shape, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.current_size = 0
        self.ptr = 0
        
        # Pre-allocate tensors
        # Assuming CIFAR-like inputs (3, 32, 32)
        # We will initialize lazily or require shape input
        self.inputs = torch.zeros((buffer_size, *input_shape), dtype=torch.float32).to(device)
        self.targets = torch.zeros((buffer_size), dtype=torch.long).to(device)
        self.task_ids = torch.zeros((buffer_size), dtype=torch.long).to(device) # Optional but useful

    def add(self, inputs, targets, task_id=None):
        n = inputs.shape[0]
        indices = torch.arange(self.ptr, self.ptr + n, device=self.device) % self.buffer_size
        
        self.inputs[indices] = inputs.to(self.device)
        self.targets[indices] = targets.to(self.device)
        if task_id is not None:
             self.task_ids[indices] = task_id
        
        self.ptr = (self.ptr + n) % self.buffer_size
        self.current_size = min(self.current_size + n, self.buffer_size)

    def sample(self, batch_size):
        if self.current_size == 0:
            return None, None

        candidate_positions = torch.arange(self.current_size, device=self.device)
        candidate_task_ids = self.task_ids[: self.current_size]
        unique_tasks = torch.unique(candidate_task_ids)
        n_tasks = int(unique_tasks.numel())

        if n_tasks <= 1:
            chosen = candidate_positions[torch.randint(0, candidate_positions.numel(), (batch_size,), device=self.device)]
            return self.inputs[chosen], self.targets[chosen]

        # Sample (approximately) equally from each task present in the buffer.
        # If a task has fewer than the requested quota, we sample with replacement.
        tasks_shuffled = unique_tasks[torch.randperm(n_tasks, device=self.device)]
        base = batch_size // n_tasks
        remainder = batch_size % n_tasks

        sampled_positions = []
        for i, task_id in enumerate(tasks_shuffled):
            n_from_task = base + (1 if i < remainder else 0)
            if n_from_task == 0:
                continue
            task_positions = candidate_positions[candidate_task_ids == task_id]
            if task_positions.numel() == 0:
                continue
            if task_positions.numel() >= n_from_task:
                perm = torch.randperm(task_positions.numel(), device=self.device)[:n_from_task]
                chosen = task_positions[perm]
            else:
                rep = torch.randint(0, task_positions.numel(), (n_from_task,), device=self.device)
                chosen = task_positions[rep]
            sampled_positions.append(chosen)

        if not sampled_positions:
            indices = candidate_positions[torch.randint(0, candidate_positions.numel(), (batch_size,), device=self.device)]
        else:
            indices = torch.cat(sampled_positions, dim=0)
            if indices.numel() < batch_size:
                extra = candidate_positions[
                    torch.randint(0, candidate_positions.numel(), (batch_size - indices.numel(),), device=self.device)
                ]
                indices = torch.cat([indices, extra], dim=0)
            elif indices.numel() > batch_size:
                indices = indices[:batch_size]

        return self.inputs[indices], self.targets[indices]
