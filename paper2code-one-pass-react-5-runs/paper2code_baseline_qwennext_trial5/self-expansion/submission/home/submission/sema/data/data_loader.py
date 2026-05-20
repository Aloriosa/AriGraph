import torch
import numpy as np
import os

class DataLoader:
    """
    Data loader for continual learning
    """
    def __init__(self, dataset_name='cifar100', data_dir='./data', batch_size=32, num_workers=4):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize dataset
        self.dataset = self._get_dataset()
        
        # Task structure for CIL
        self.tasks = []
        self.current_task = 0
        
    def _get_dataset(self):
        """Get dataset based on name"""
        if self.dataset_name == 'cifar100':
            from data.dataset import CIFAR100Dataset
            return CIFAR100Dataset(self.data_dir, self.batch_size, self.num_workers)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
    def setup_tasks(self, num_tasks=20):
        """Setup tasks for CIL"""
        if self.dataset_name == 'cifar100':
            # 100 classes, 5 classes per task
            classes = list(range(100))
            np.random.shuffle(classes)
            self.tasks = [classes[i:i+5] for i in range(0, 100, 5)]
            print(f"Setup {len(self.tasks)} tasks with 5 classes each")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
    def get_current_task(self):
        """Get current task"""
        return self.tasks[self.current_task]
        
    def get_train_loader(self, classes, shuffle=True):
        """Get training loader for current task"""
        return self.dataset.get_train_loader(classes, shuffle)
        
    def get_test_loader(self, classes, shuffle=False):
        """Get test loader for current task"""
        return self.dataset.get_test_loader(classes, shuffle)
        
    def next_task(self):
        """Next task"""
        self.current_task += 1
        return self.current_task < len(self.tasks)
        
    def reset(self):
        """Reset to first task"""
        self.current_task = 0