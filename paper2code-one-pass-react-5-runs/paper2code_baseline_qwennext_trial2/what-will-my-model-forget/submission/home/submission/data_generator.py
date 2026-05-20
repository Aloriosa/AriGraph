import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

class ForecastingDataset(Dataset):
    """
    Dataset for forecasting forgotten examples
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_dataset():
    """
    Generate a simulated dataset similar to the P3 dataset described in the paper.
    This is a simplified version with 1000 examples for training, 200 for validation, and 200 for testing.
    """
    np.random.seed(42)
    
    # Simulate the dataset with 1000 examples for training, 200 for validation, and 200 for testing
    # Each example has 768-dimensional representations (similar to BERT embeddings)
    train_data = np.random.randn(1000, 768)
    train_labels = np.random.randint(0, 2, 1000)
    
    val_data = np.random.randn(200, 768)
    val_labels = np.random.randint(0, 2, 200)
    
    test_data = np.random.randn(200, 768)
    test_labels = np.random.randint(0, 2, 200)
    
    # Create datasets
    train_dataset = ForecastingDataset(train_data, train_labels)
    val_dataset = ForecastingDataset(val_data, val_labels)
    test_dataset = ForecastingDataset(test_data, test_labels)
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    np.save('data/train_data.npy', train_data)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/val_data.npy', val_data)
    np.save('data/val_labels.npy', val_labels)
    np.save('data/test_data.npy', test_data)
    np.save('data/test_labels.npy', test_labels)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    generate_dataset()