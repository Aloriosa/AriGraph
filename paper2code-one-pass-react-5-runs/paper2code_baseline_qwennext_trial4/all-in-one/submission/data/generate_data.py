"""
Generate synthetic data for benchmark tasks
"""
import numpy as np
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_gaussian_linear_data(n_samples=10000, n_dim=10, noise_std=0.1):
    """
    Generate data for Gaussian Linear task.
    
    Parameters:
    n_samples: Number of samples to generate
    n_dim: Dimensionality of parameters and data
    noise_std: Standard deviation of noise
    
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from normal distribution
    parameters = np.random.normal(0, np.sqrt(0.1), (n_samples, n_dim))
    
    # Generate data from parameters
    data = parameters + np.random.normal(0, noise_std, (n_samples, n_dim))
    
    return parameters, data

def generate_gaussian_mixture_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for Gaussian Mixture task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of noise
    
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from uniform distribution
    parameters = np.random.uniform(-10, 10, (n_samples, 2))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 2))
    for i in range(n_samples):
        if np.random.random() < 0.5:
            data[i] = parameters[i] + np.random.normal(0, 1, 2)
        else:
            data[i] = parameters[i] + np.random.normal(0, 0.01, 2)
    
    return parameters, data

def generate_two_moons_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for Two Moons task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of data
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from uniform distribution
    parameters = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 2))
    for i in range(n_samples):
        alpha = np.random.uniform(-np.pi/2, np.pi/2)
        r = np.random.normal(0.1, 0.01)
        data[i, 0] = r * np.cos(alpha) + 0.25
        data[i, 1] = r * np.sin(alpha)
        
        # Add rotation
        theta = np.arctan2(parameters[i, 1], parameters[i, 0])
        data[i, 0] = data[i, 0] * np.cos(theta) - data[i, 1] * np.sin(theta)
        data[i, 1] = data[i, 0] * np.sin(theta) + data[i, 1] * np.cos(theta)
    
    return parameters, data

def generate_slcp_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for SLCP task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of data
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from uniform distribution
    parameters = np.random.uniform(-3, 3, (n_samples, 5))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 8))
    for i in range(n_samples):
        # Generate 4 data points from normal distribution
        for j in range(4):
            mu = np.array([parameters[i, 0], parameters[i, 1]])
            cov = np.array([[parameters[i, 2]**2, np.tanh(parameters[i, 4]) * parameters[i, 2] * parameters[i, 3]],
                            [np.tanh(parameters[i, 4]) * parameters[i, 2] * parameters[i, 3], parameters[i, 3]**2]])
            data[i, j*2:(j+1)*2] = np.random.multivariate_normal(mu, cov)
    
    return parameters, data

def generate_lotka_volterra_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for Lotka-Volterra task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of data
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from sigmoid-transformed normal distribution
    parameters = np.random.normal(0, 1, (n_samples, 4))
    parameters = 1 + 2 / (1 + np.exp(-parameters))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 100))
    for i in range(n_samples):
        # Solve ODEs
        dt = 0.1
        x = np.zeros(100)
        y = np.zeros(100)
        x[0] = 1
        y[0] = 1
        for j in range(1, 100):
            x[j] = x[j-1] + dt * (parameters[i, 0] * x[j-1] - parameters[i, 1] * x[j-1] * y[j-1])
            y[j] = y[j-1] + dt * (parameters[i, 2] * x[j-1] * y[j-1] - parameters[i, 3] * y[j-1])
        data[i, :] = x
    
    return parameters, data

def generate_sird_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for SIRD task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of data
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from uniform distribution
    parameters = np.random.uniform(0, 0.5, (n_samples, 3))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 100))
    for i in range(n_samples):
        # Solve ODEs
        dt = 0.1
        s = np.zeros(100)
        i = np.zeros(100)
        r = np.zeros(100)
        d = np.zeros(100)
        s[0] = 1
        i[0] = 0.01
        r[0] = 0
        d[0] = 0
        for j in range(1, 100):
            s[j] = s[j-1] - dt * parameters[i, 0] * s[j-1] * i[j-1]
            i[j] = i[j-1] + dt * (parameters[i, 0] * s[j-1] * i[j-1] - parameters[i, 1] * i[j-1] - parameters[i, 2] * i[j-1])
            r[j] = r[j-1] + dt * parameters[i, 1] * i[j-1]
            d[j] = d[j-1] + dt * parameters[i, 2] * i[j-1]
        data[i, :] = i
    
    return parameters, data

def generate_hodgkin_huxley_data(n_samples=10000, noise_std=0.1):
    """
    Generate data for Hodgkin-Huxley task.
    
    Parameters:
    n_samples: Number of samples to generate
    noise_std: Standard deviation of data
    Returns:
    parameters: Parameters array
    data: Data array
    """
    # Sample parameters from uniform distribution
    parameters = np.random.uniform(0, 1, (n_samples, 7))
    
    # Generate data from parameters
    data = np.zeros((n_samples, 100))
    for i in range(n_samples):
        # Solve ODEs
        dt = 0.1
        v = np.zeros(100)
        m = np.zeros(100)
        h = np.zeros(100)
        n = np.zeros(100)
        v[0] = -65
        m[0] = 0.01
        h[0] = 0.01
        n[0] = 0.01
        for j in range(1, 100):
            v[j] = v[j-1] + dt * (parameters[i, 0] * (parameters[i, 1] * m[j-1]**3 * h[j-1] * (parameters[i, 2] - v[j-1]) + parameters[i, 3] * n[j-1]**4 * (parameters[i, 4] - v[j-1]) + parameters[i, 5] * (parameters[i, 6] - v[j-1]) + parameters[i, 7]))
            m[j] = m[j-1] + dt * (parameters[i, 0] * (1 - m[j-1]) - parameters[i, 1] * m[j-1])
            h[j] = h[j-1] + dt * (parameters[i, 2] * (1 - h[j-1]) - parameters[i, 3] * h[j-1])
            n[j] = n[j-1] + dt * (parameters[i, 4] * (1 - n[j-1]) - parameters[i, 5] * n[j-1])
        data[i, :] = v
    
    return parameters, data

def main():
    """Main function."""
    # Generate data for benchmark tasks
    logger.info("Generating data for benchmark tasks...")
    parameters, data = generate_gaussian_linear_data()
    np.save("data/gaussian_linear_parameters.npy", parameters)
    np.save("data/gaussian_linear_data.npy", data)
    
    parameters, data = generate_gaussian_mixture_data()
    np.save("data/gaussian_mixture_parameters.npy", data)
    np.save("data/gaussian_mixture_data.npy", data)
    
    parameters, data = generate_two_moons_data()
    np.save("data/two_moons_parameters.npy", parameters)
    np.save("data/two_moons_data.npy", data)
    
    parameters, data = generate_slcp_data()
    np.save("data/slcp_parameters.npy", parameters)
    np.save("data/slcp_data.npy", data)
    
    # Generate data for other tasks
    logger.info("Generating data for other tasks...")
    parameters, data = generate_lotka_volterra_data()
    np.save("data/lotka_volterra_parameters.npy", parameters)
    np.save("data/lotka_volterra_data.npy", data)
    
    parameters, data = generate_sird_data()
    np.save("data/sird_parameters.npy", parameters)
    np.save("data/sird_data.npy", data)
    
    parameters, data = generate_hodgkin_huxley_data()
    np.save("data/hodgkin_huxley_parameters.npy", parameters)
    np.save("data/hodgkin_huxley_data.npy", data)
    
    logger.info("Data generation complete.")

if __name__ == "__main__":
    main()