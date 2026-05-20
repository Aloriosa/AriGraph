"""
Data generation and loading for Simformer
"""
import numpy as np
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(task: str = "gaussian_linear", n_samples: int = 10000) -> tuple:
    """
    Load data for a specific task.
    
    Parameters:
    task: Name of the task
    n_samples: Number of samples to load
    Returns:
    parameters: Parameters array
    data: Data array
    """
    data_dir = "data"
    
    if task == "gaussian_linear":
        parameters = np.load(os.path.join(data_dir, "gaussian_linear_parameters.npy"))
        data = np.load(os.path.join(data_dir, "gaussian_linear_data.npy"))
    elif task == "gaussian_mixture":
        parameters = np.load(os.path.join(data_dir, "gaussian_mixture_parameters.npy"))
        data = np.load(os.path.join(data_dir, "gaussian_mixture_data.npy"))
    elif task == "two_moons":
        parameters = np.load(os.path.join(data_dir, "two_moons_parameters.npy"))
        parameters = np.random.uniform(-1, 1, (n_samples, 2))
        data = np.load(os.path.join(data_dir, "two_moons_data.npy"))
    elif task == "slcp":
        parameters = np.load(os.path.join(data_dir, "slcp_parameters.npy"))
        data = np.load(os.path.join(data_dir, "slcp_data.npy"))
    elif task == "lotka_volterra":
        parameters = np.load(os.path.join(data_dir, "lotka_volterra_parameters.npy"))
        data = np.load(os.path_dir, "lotka_volterra_data.npy")
    elif task == "sird":
        parameters = np.load(os.path.join(data_dir, "sird_parameters.npy"))
        data = np.load(os.path.join(data_dir, "sird_data.npy"))
    elif task == "hodgkin_huxley":
        parameters = np.load(os.path.join(data_dir, "hodgkin_huxley_parameters.npy"))
        data = np.load(os.path.join(data_dir, "hodgkin_huxley_data.npy"))
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return parameters, data

def generate_data(task: str = "gaussian_linear", n_samples: int = 10000) -> tuple:
    """
    Generate data for a specific task.
    
    Parameters:
    task: Name of the task
    n_samples: Number of samples to generate
    Returns:
    parameters: Parameters array
    data: Data array
    """
    if task == "gaussian_linear":
        parameters, data = generate_gaussian_linear_data(n_samples)
    elif task == "gaussian_mixture":
        parameters, data = generate_gaussian_mixture_data(n_samples)
    elif task == "two_moons":
        parameters, data = generate_two_moons_data(n_samples)
    elif task == "slcp":
        parameters, data = generate_slcp_data(n_samples)
    elif task == "lotka_volterra":
        parameters, data = generate_lotka_volterra_data(n_samples)
    elif task == "sird":
        parameters, data = generate_sird_data(n_samples)
    elif task == "hodgkin_huxley":
        parameters, data = generate_hodgkin_huxley_data(n_samples)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return parameters, data

def save_data(task: str, parameters: np.ndarray, data: np.ndarray) -> None:
    """
    Save data to files.
    
    Parameters:
    task: Name of the task
    parameters: Parameters array
    data: Data array
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, f"{task}_parameters.npy"), parameters)
    np.save(os.path.join(data_dir, f"{task}_data.npy"), data)

def main():
    """Main function."""
    # Generate and save data for all tasks
    tasks = ["gaussian_linear", "gaussian_mixture", "two_moons", "slcp", "lotka_volterra", "sird", "hodgkin_huxley"]
    for task in tasks:
        logger.info(f"Generating data for {task}...")
        parameters, data = generate_data(task)
        save_data(task, parameters, data)
    logger.info("Data generation complete.")

if __name__ == "__main__":
    main()