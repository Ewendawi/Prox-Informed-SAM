import os
import torch
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import torchvision.transforms as transforms

def get_split_cifar100(n_tasks=20, seed=42, data_root='./data', max_train_per_task=None, max_test_per_task=None):
    cache_name = f'split_cifar100_{n_tasks}tasks_seed{seed}_train{max_train_per_task}_test{max_test_per_task}.pt'
    cache_path = os.path.join(data_root, cache_name)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Download check
    train_set = CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    test_set = CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

    print(f"train_set size {len(train_set)}")
    print(f"test_set size {len(test_set)}")
    
    # Try Loading Cache
    if os.path.exists(cache_path):
        print(f"Loading split from {cache_path}")
        split_info = torch.load(cache_path, weights_only=False)
        train_indices_list = split_info['train']
        test_indices_list = split_info['test']
        
        tasks_train = [Subset(train_set, idx) for idx in train_indices_list]
        tasks_test = [Subset(test_set, idx) for idx in test_indices_list]
        return tasks_train, tasks_test
        
    print(f"Creating new split for {n_tasks} tasks (seed {seed})")
    # Split classes
    rng = np.random.RandomState(seed)
    classes = np.arange(100)
    # rng.shuffle(classes) # Keep fixed for standard Benchmark
    
    per_task = 100 // n_tasks
    train_indices_list = []
    test_indices_list = []
    tasks_train = []
    tasks_test = []
    
    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set.targets)
    
    for t in range(n_tasks):
        c_start = t * per_task
        c_end = (t + 1) * per_task
        task_classes = classes[c_start:c_end]
        
        # Filter indices
        train_idx = np.where(np.isin(train_targets, task_classes))[0]
        test_idx = np.where(np.isin(test_targets, task_classes))[0]
        
        if max_train_per_task is not None and len(train_idx) > max_train_per_task:
            train_idx = rng.choice(train_idx, size=max_train_per_task, replace=False)
        if max_test_per_task is not None and len(test_idx) > max_test_per_task:
            test_idx = rng.choice(test_idx, size=max_test_per_task, replace=False)

        print(f"train_idx size {len(train_idx)}")
        print(f"test_idx size {len(test_idx)}")

        train_indices_list.append(train_idx)
        test_indices_list.append(test_idx)
        
        tasks_train.append(Subset(train_set, train_idx))
        tasks_test.append(Subset(test_set, test_idx))

    print(f"train_indices_list size {len(train_indices_list)}")
    print(f"test_indices_list size {len(test_indices_list)}")
    
    # Save Cache
    print(f"Saving split to {cache_path}")
    torch.save({'train': train_indices_list, 'test': test_indices_list}, cache_path)
        
    return tasks_train, tasks_test
