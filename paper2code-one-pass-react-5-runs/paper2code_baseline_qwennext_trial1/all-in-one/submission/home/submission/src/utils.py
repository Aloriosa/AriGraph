import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def load_data(data_path):
    """
    Load data from file
    """
    if data_path.endswith('.npy'):
        data = np.load(data_path)
    elif data_path.endswith('.pt'):
        data = torch.load(data_path)
    else:
        raise ValueError("Unsupported file format")
    return data

def save_data(data, save_path):
    """
    Save data to file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(data, np.ndarray):
        np.save(save_path, data)
    elif isinstance(data, torch.Tensor):
        torch.save(data, save_path)
    else:
        raise ValueError("Unsupported data type")

def plot_results(data, title="Results", save_path=None):
    """
    Plot results
    """
    plt.figure(figsize=(10, 5))
    if len(data.shape) == 1:
        plt.plot(data)
    else:
        for i in range(min(data.shape[1], 5)):
            plt.plot(data[:, i], label=f"Dimension {i}")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    corr = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    
    return {
        "MSE": mse,
        "MAE": mae,
        "Correlation": corr
    }

def main():
    """
    Main function
    """
    print("Running utility functions...")
    
    # Test data loading
    print("Testing data loading...")
    data = np.random.randn(100, 5)
    save_data(data, "/tmp/test_data.npy")
    loaded_data = load_data("/tmp/test_data.npy")
    print(f"Loaded data shape: {loaded_data.shape}")
    
    # Test plotting
    print("Testing plotting...")
    plot_results(data, "Test Plot", "/tmp/test_plot.png")
    
    # Test metrics
    print("Testing metrics calculation...")
    metrics = calculate_metrics(data, data)
    print(f"Metrics: {metrics}")
    
    print("Utility functions completed!")

if __name__ == "__main__":
    main()