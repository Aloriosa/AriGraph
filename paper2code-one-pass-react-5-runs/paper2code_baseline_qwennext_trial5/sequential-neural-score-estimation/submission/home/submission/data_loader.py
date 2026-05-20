"""
Data loader for simulation-based inference tasks.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional

class SBIDataset(Dataset):
    """
    Dataset class for simulation-based inference tasks.
    """
    
    def __init__(self, data_path, mode='train'):
        super(SBIDataset, self).__init__()
        
        self.mode = mode
        self.data_path = data_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load data from file"""
        if os.path.exists(self.data_path):
            self.data = np.load(self.data_path)
            if 'theta' in self.data and 'x' in self.data:
                self.theta = self.data['theta']
                self.x = self.data['x']
        else:
            # If no data is available, create synthetic data
            self.create_synthetic_data()
            
    def create_synthetic_data(self):
        """Create synthetic data for testing"""
        # Create synthetic data based on the benchmark tasks
        np.random.seed(42)
        
        # Define parameters for synthetic data
        theta_dim = 10
        x_dim = 10
        n_samples = 1000
        
        # Generate synthetic data for Gaussian Linear benchmark
        theta = np.random.randn(n_samples, theta_dim) * 0.1
        x = np.random.randn(n_samples, x_dim) * 0.1
        
        # Add some correlation between theta and x
        for i in range(n_samples):
            x[i] = theta[i] + np.random.randn(x_dim) * 0.1
        
        self.theta = theta
        self.x = x
        
    def __len__(self):
        return len(self.theta)
    
    def __getitem__(self, idx):
        theta = torch.tensor(self.theta[idx], dtype=torch.float32)
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        
        return theta, x