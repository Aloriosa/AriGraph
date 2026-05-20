import numpy as np
from scipy.linalg import inv, sqrtm
from utils import kl_gaussian, target_gaussian_score

class BaM:
    """
    Batch and Match (BaM) variational inference for Gaussian targets.
    """
    def __init__(self, target_mu, target_Sigma, batch_size=10, lambda_reg=1.0,
                 max_iter=2000, seed=0):
        self.D = target_mu.shape[0]
        self.mu_target = target_mu
        self.Sigma_target = target_Sigma
        self.Sigma_target_inv = inv(target_Sigma)
        self.B = batch_size
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        # initialise variational parameters
        self.mu = np.zeros_like(target_mu)
        self.Sigma = np.eye(self.D)

    def _sample_q(self, n):
        return self.rng.multivariate_normal(self.mu, self.Sigma, size=n)

    def _update(self):
        B = self.B
        mu = self.mu
        Sigma = self.Sigma
        lambda_reg = self.lambda_reg

        # sample from current q
        z_samples = self._sample_q(B)

        # target scores
        g_samples = np.array([target_gaussian_score(z, self.mu_target, self.Sigma_target_inv)
                              for z in z_samples])

        # statistics
        bar_z = np.mean(z_samples, axis=0)
        C = np.cov(z_samples, rowvar=False, bias=True)  # unbiased B in denom
        bar_g = np.mean(g_samples, axis=0)
        Gamma = np.cov(g_samples, rowvar=False, bias=True)

        # update covariance
        U = lambda_reg * Gamma + (lambda_reg / (1 + lambda_reg)) * np.outer(bar_g, bar_g)
        V = Sigma + lambda_reg * C + (lambda_reg / (1 + lambda_reg)) * np.outer(mu - bar_z, mu - bar_z)

        # compute sqrtm(I + 4UV)
        sqrt_term = sqrtm(np.eye(self.D) + 4 * U @ V)
        Sigma_new = 2 * V @ np.linalg.inv(np.eye(self.D) + sqrt_term)

        # update mean
        mu_new = (1 / (1 + lambda_reg)) * mu + (lambda_reg / (1 + lambda_reg)) * (Sigma_new @ bar_g + bar_z)

        self.mu, self.Sigma = mu_new, Sigma_new

    def run(self):
        """
        Run BaM for max_iter iterations.
        Returns:
            kl_history: list of KL(q||p) values after each iteration
            grad_evals: total number of target score evaluations (B * iter)
        """
        kl_history = []
        for t in range(self.max_iter):
            self._update()
            kl = kl_gaussian(self.mu, self.Sigma, self.mu_target, self.Sigma_target)
            kl_history.append(kl)
        grad_evals = self.max_iter * self.B
        return kl_history, grad_evals