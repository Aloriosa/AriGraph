import numpy as np
import torch
import torch.distributions as dist
import pickle
import csv
import os
from tqdm import tqdm

class BBVIBaseline:
    """
    ELBO-based Black-box Variational Inference (BBVI) baseline
    Uses reparameterization trick and Adam optimizer
    """
    
    def __init__(self, target_log_density, dim, batch_size=100, max_iterations=500, learning_rate=0.01):
        self.target_log_density = target_log_density
        self.dim = dim
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
        # Initialize variational parameters
        self.mean = torch.zeros(dim, requires_grad=True)
        self.log_std = torch.zeros(dim, requires_grad=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam([self.mean, self.log_std], lr=learning_rate)
        
        # Store history
        self.history = {
            'means': [],
            'log_stds': [],
            'elbos': [],
            'iterations': []
        }
    
    def elbo(self, samples):
        """
        Compute ELBO: E_q[log p(x)] - E_q[log q(x)]
        """
        # Compute log p(x)
        log_p = self.target_log_density(samples)
        
        # Compute log q(x) for Gaussian
        std = torch.exp(self.log_std)
        q_dist = dist.Normal(self.mean, std)
        log_q = q_dist.log_prob(samples).sum(dim=1)
        
        # ELBO = E_q[log p(x) - log q(x)]
        elbo = (log_p - log_q).mean()
        return elbo
    
    def fit(self, verbose=True):
        if verbose:
            pbar = tqdm(range(self.max_iterations), desc="BBVI Iterations")
        else:
            pbar = range(self.max_iterations)
            
        for i in pbar:
            # Sample from variational distribution
            std = torch.exp(self.log_std)
            q_dist = dist.Normal(self.mean, std)
            samples = q_dist.sample((self.batch_size,))
            
            # Compute ELBO
            elbo = self.elbo(samples)
            
            # Maximize ELBO (minimize -ELBO)
            self.optimizer.zero_grad()
            (-elbo).backward()
            self.optimizer.step()
            
            # Store history
            self.history['means'].append(self.mean.detach().clone())
            self.history['log_stds'].append(self.log_std.detach().clone())
            self.history['elbos'].append(elbo.item())
            self.history['iterations'].append(i)
            
            if verbose:
                pbar.set_postfix({'elbo': f'{elbo.item():.6f}'})
                
            # Early stopping
            if len(self.history['elbos']) > 1:
                if abs(self.history['elbos'][-2] - self.history['elbos'][-1]) < 1e-6:
                    if verbose:
                        print(f"\nConverged at iteration {i+1}")
                    break
                    
        return self.history
    
    def get_final_parameters(self):
        std = torch.exp(self.log_std)
        return self.mean.detach().clone(), torch.diag(std ** 2)

def create_comparison_results():
    """
    Compare BAM with BBVI baseline on different target distributions
    """
    dim = 10
    batch_size = 100
    max_iterations = 50  # BAM iterations
    bbvi_iterations = 500  # BBVI iterations (more iterations needed)
    
    targets = {
        'gaussian': lambda x: -0.5 * torch.sum((x - torch.tensor([1.0]*5 + [0.0]*5)) ** 2, dim=1),
        'mixture': lambda x: torch.logsumexp(
            torch.stack([
                -0.5 * torch.sum((x - torch.tensor([2.0]*5 + [0.0]*5)) ** 2, dim=1),
                -0.5 * torch.sum((x - torch.tensor([-2.0]*5 + [0.0]*5)) ** 2, dim=1)
            ]), dim=0) - np.log(2),
        'hierarchical': lambda x: torch.tensor(0.0)  # Will be replaced with actual function
    }
    
    # For hierarchical model, create a synthetic dataset
    n_obs = 50
    X = torch.randn(n_obs, dim)
    true_beta = torch.randn(dim) * 0.5
    y = torch.matmul(X, true_beta) + torch.randn(n_obs) * 0.1
    
    def hierarchical_target(x):
        # Prior: β ~ N(0, I)
        prior_log_prob = -0.5 * torch.sum(x ** 2, dim=1)
        
        # Likelihood: y ~ N(Xβ, σ²I)
        likelihood_log_prob = -0.5 * torch.sum((y - torch.matmul(X, x.T)) ** 2, dim=1)
        
        return prior_log_prob + likelihood_log_prob
    
    targets['hierarchical'] = hierarchical_target
    
    results = []
    
    for target_name, target_func in targets.items():
        print(f"Comparing BAM and BBVI on {target_name} target...")
        
        # Run BAM
        bam = BAMAlgorithm(
            target_log_density=target_func,
            dim=dim,
            batch_size=batch_size,
            max_iterations=max_iterations,
            regularization=1e-6
        )
        bam.fit(verbose=False)
        bam_mean, bam_cov = bam.get_final_parameters()
        
        # Run BBVI
        bbvi = BBVIBaseline(
            target_log_density=target_func,
            dim=dim,
            batch_size=batch_size,
            max_iterations=bbvi_iterations,
            learning_rate=0.01
        )
        bbvi.fit(verbose=False)
        bbvi_mean, bbvi_cov = bbvi.get_final_parameters()
        
        # Compute ground truth
        if target_name == 'gaussian':
            true_mean = np.array([1.0]*5 + [0.0]*5)
            true_cov = np.eye(dim)
        elif target_name == 'mixture':
            true_mean = np.zeros(dim)
            true_cov = np.eye(dim)
            true_cov[:dim//2, :dim//2] += 4.0
        else:  # hierarchical
            XtX = torch.matmul(X.T, X)
            XtY = torch.matmul(X.T, y)
            true_cov = torch.inverse(XtX + torch.eye(dim)).numpy()
            true_mean = torch.matmul(true_cov, XtY).numpy()
        
        # Compute errors
        bam_mean_error = np.linalg.norm(bam_mean.numpy() - true_mean)
        bbvi_mean_error = np.linalg.norm(bbvi_mean.numpy() - true_mean)
        
        bam_cov_error = np.linalg.norm(bam_cov.numpy() - true_cov)
        bbvi_cov_error = np.linalg.norm(bbvi_cov.numpy() - true_cov)
        
        # Compute score divergence for BAM
        samples = dist.MultivariateNormal(bam_mean, bam_cov).sample((1000,))
        target_scores = torch.autograd.grad(target_func(samples).sum(), samples, create_graph=True)[0]
        variational_scores = -torch.matmul(torch.inverse(bam_cov + 1e-6 * torch.eye(dim)), (samples - bam_mean).T).T
        bam_score_divergence = torch.mean(torch.sum((target_scores - variational_scores) ** 2, dim=1)).item()
        
        # Compute ELBO for BBVI
        bbvi_samples = dist.MultivariateNormal(bbvi_mean, bbvi_cov).sample((1000,))
        bbvi_elbo = -bbvi.elbo(bbvi_samples).item()
        
        # Store results
        results.append({
            'target': target_name,
            'bam_mean_error': bam_mean_error,
            'bbvi_mean_error': bbvi_mean_error,
            'bam_cov_error': bam_cov_error,
            'bbvi_cov_error': bbvi_cov_error,
            'bam_score_divergence': bam_score_divergence,
            'bbvi_elbo': bbvi_elbo,
            'bam_iterations': max_iterations,
            'bbvi_iterations': bbvi_iterations
        })
    
    # Save results
    output_file = 'results/baseline_comparison.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'target', 'bam_mean_error', 'bbvi_mean_error', 'bam_cov_error', 'bbvi_cov_error',
            'bam_score_divergence', 'bbvi_elbo', 'bam_iterations', 'bbvi_iterations'
        ])
        
        for result in results:
            writer.writerow([
                result['target'], result['bam_mean_error'], result['bbvi_mean_error'],
                result['bam_cov_error'], result['bbvi_cov_error'],
                result['bam_score_divergence'], result['bbvi_elbo'],
                result['bam_iterations'], result['bbvi_iterations']
            ])
    
    print(f"Baseline comparison results saved to {output_file}")

def main():
    create_comparison_results()

if __name__ == "__main__":
    main()