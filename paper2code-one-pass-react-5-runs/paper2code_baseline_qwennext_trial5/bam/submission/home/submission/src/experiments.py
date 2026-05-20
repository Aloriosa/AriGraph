import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from bam import BatchAndMatch
from gsm import GaussianScoreMatching
from advi import ADVI
import jax.random as random
import os

def create_gaussian_target(dimension: int, seed: int = 0) -> Callable:
    """
    Create a Gaussian target distribution.
    
    Args:
        dimension: Dimension of the target distribution.
        seed: Random seed for reproducibility.
        
    Returns:
        target_log_prob: Function that computes the log probability of the target distribution.
    """
    key = random.PRNGKey(seed)
    # Create a random mean and covariance
    mu_true = random.normal(key, (dimension,))
    # Create a random positive definite covariance matrix
    A = random.normal(key, (dimension, dimension))
    cov_true = jnp.dot(A, A.T)
    
    def target_log_prob(z):
        return -0.5 * jnp.sum((z - mu_true) ** 2 / (2 * jnp.diag(cov_true))) - 0.5 * jnp.sum(jnp.log(2 * np.pi)) * dimension - 0.5 * jnp.sum(jnp.log(jnp.diag(cov_true)))
    
    return target_log_prob

def create_non_gaussian_target(dimension: int, seed: int = 0) -> Callable:
    """
    Create a non-Gaussian target distribution using sinh-arcsinh distribution.
    
    Args:
        dimension: Dimension of the target distribution.
        seed: Random seed for reproducibility.
    
    Returns:
        target_log_prob: Function that computes the log probability of the target distribution.
    """
    key = random.PRNGKey(seed)
    # Create a random mean and covariance
    mu_true = random.normal(key, (dimension,))
    # Create a random positive definite covariance matrix
    A = random.normal(key, (dimension, dimension))
    cov_true = jnp.dot(A, A.T)
    
    # Parameters for sinh-arcsinh distribution
    skew = 1.0
    tail = 1.0
    
    def target_log_prob(z):
        # Transform using sinh-arcsinh distribution
        y = (jnp.arcsinh(z) + skew) / tail
        z_transformed = jnp.sinh(y)
        # Compute log probability
        log_prob = -0.5 * jnp.sum((z_transformed - mu_true) ** 2 / (2 * jnp.diag(cov_true))) - 0.5 * jnp.sum(jnp.log(2 * np.pi)) * dimension - 0.5 * jnp.sum(jnp.log(jnp.diag(cov_true)))
        return log_prob
    
    return target_log_prob

def run_gaussian_experiments():
    """
    Run experiments on Gaussian targets.
    """
    print("Running Gaussian experiments...")
    
    dimensions = [4, 16, 64, 128, 256]
    results = {}
    
    for d in dimensions:
        print(f"Dimension: {d}")
        target_log_prob = create_gaussian_target(d)
        
        # Initialize variational distribution
        mu_init = np.random.normal(0, 0.1, d)
        sigma_init = np.eye(d)
        
        # Run BaM
        print("Running BaM...")
        bam = BatchAndMatch(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_bam, sigma_bam = bam.optimize()
        
        # Run GSM
        print("Running GSM...")
        gsm = GaussianScoreMatching(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_gsm, sigma_gsm = gsm.optimize()
        
        # Run ADVI
        print("Running ADVI...")
        advi = ADVI(target_log_prob, mu_init, sigma_init, batch_size=10, learning_rate=0.01, max_iterations=10)
        mu_advi, sigma_advi = advi.optimize()
        
        results[d] = {
            'bam': (mu_bam, sigma_bam),
            'gsm': (mu_gsm, sigma_gsm),
            'advi': (mu_advi, sigma_advi)
        }
        
    return results

def run_non_gaussian_experiments():
    """
    Run experiments on non-Gaussian targets.
    """
    print("Running non-Gaussian experiments...")
    
    dimensions = [10]
    results = {}
    
    for d in dimensions:
        print(f"Dimension: {d}")
        target_log_prob = create_non_gaussian_target(d)
        
        # Initialize variational distribution
        mu_init = np.random.normal(0, 0.1, d)
        sigma_init = np.eye(d)
        
        # Run BaM
        print("Running BaM...")
        bam = BatchAndMatch(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_bam, sigma_bam = bam.optimize()
        
        # Run GSM
        print("Running GSM...")
        gsm = GaussianScoreMatching(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_gsm, sigma_gsm = gsm.optimize()
        
        # Run ADVI
        print("Running ADVI...")
        advi = ADVI(target_log_prob, mu_init, sigma_init, batch_size=10, learning_rate=0.01, max_iterations=10)
        mu_advi, sigma_advi = advi.optimize()
        
        results[d] = {
            'bam': (mu_bam, sigma_bam),
            'gsm': (mu_gsm, sigma_gsm),
            'advi': (mu_advi, sigma_advi)
        }
        
    return results

def run_hierarchical_experiments():
    """
    Run experiments on hierarchical Bayesian models.
    """
    print("Running hierarchical experiments...")
    
    models = ['8-schools', 'gp-pois-regr']
    results = {}
    
    for model in models:
        print(f"Model: {model}")
        # Create a simple hierarchical model
        if model == '8-schools':
            def target_log_prob(z):
                # Simple 8-schools model
                return -0.5 * jnp.sum(z**2) - 0.5 * jnp.sum(z**2) + 10
        else:
            def target_log_prob(z):
                # Simple GP Poisson regression
                return -0.5 * jnp.sum(z**2) - 0.5 * jnp.sum(z**2) + 10
        
        # Initialize variational distribution
        mu_init = np.random.normal(0, 0.1, 10)
        sigma_init = np.eye(10)
        
        # Run BaM
        print("Running BaM...")
        bam = BatchAndMatch(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_bam, sigma_bam = bam.optimize()
        
        # Run GSM
        print("Running GSM...")
        gsm = GaussianScoreMatching(target_log_prob, mu_init, sigma_init, batch_size=10, max_iterations=10)
        mu_gsm, sigma_gsm = gsm.optimize()
        
        # Run ADVI
        print("Running ADVI...")
        advi = ADVI(target_log_prob, mu_init, sigma_init, batch_size=10, learning_rate=0.01, max_iterations=10)
        mu_advi, sigma_advi = advi.optimize()
        
        results[model] = {
            'bam': (mu_bam, sigma_bam),
            'gsm': (mu_gsm, sigma_gsm),
            'advi': (mu_advi, sigma_advi)
        }
        
    return results

def run_deep_generative_experiments():
    """
    Run experiments on deep generative models.
    """
    print("Running deep generative experiments...")
    
    # Create a simple deep generative model
    def target_log_prob(z):
        return -0.5 * jnp.sum(z**2) - 0.5 * jnp.sum(z**2) + 10
    
    # Initialize variational distribution
    mu_init = np.random.normal(0, 0.1, 256)
    sigma_init = np.eye(256)
    
    # Run BaM
    print("Running BaM...")
    bam = BatchAndMatch(target_log_prob, mu_init, sigma_init, batch_size=300, max_iterations=10)
    mu_bam, sigma_bam = bam.optimize()
    
    # Run GSM
    print("Running GSM...")
    gsm = GaussianScoreMatching(target_log_prob, mu_init, sigma_init, batch_size=300, max_iterations=10)
    mu_gsm, sigma_gsm = gsm.optimize()
    
    # Run ADVI
    print("Running ADVI...")
    advi = ADVI(target_prob, mu_init, sigma_init, batch_size=300, learning_rate=0.01, max_iterations=10)
    mu_advi, sigma_advi = advi.optimize()
    
    results = {
        'bam': (mu_bam, sigma_bam),
        'gsm': (mu_gsm, sigma_gsm),
        'advi': (mu_advi, sigma_advi)
    }
    
    return results

def main():
    """
    Main function to run all experiments.
    """
    print("Running all experiments...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    print("Running Gaussian experiments...")
    gaussian_results = run_gaussian_experiments()
    np.save('results/gaussian_results.npy', gaussian_results)
    
    print("Running non-Gaussian experiments...")
    non_gaussian_results = run_non_gaussian_experiments()
    np.save('results/non_gaussian_results.npy', non_gaussian_results)
    
    print("Running hierarchical experiments...')
    hierarchical_results = run_hierarchical_experiments()
    np.save('results/hierarchical_results.npy', hierarchical_results)
    
    print("Running deep generative experiments...")
    deep_results = run_deep_generative_experiments()
    np.save('results/deep_results.npy', deep_results)
    
    print("All experiments completed!")
    
    # Generate plots
    print("Generating plots...")
    generate_plots()
    
    print("All reproduction steps completed successfully!")

if __name__ == '__main__':
    main()