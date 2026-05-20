import numpy as np
from utils import kl_gaussian, target_gaussian_score, inv, det

class GSM:
    """
    Gaussian Score Matching (GSM) algorithm (Modi et al., 2023) for Gaussian target.
    Uses batch size B=1.
    """
    def __init__(self, target_mu, target_Sigma, max_iter=2000, seed=0):
        self.D = target_mu.shape[0]
        self.mu_target = target_mu
        self.Sigma_target = target_Sigma
        self.Sigma_target_inv = inv(target_Sigma)
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        # initialise variational parameters
        self.mu = np.zeros_like(target_mu)
        self.Sigma = np.eye(self.D)

    def _update(self):
        mu = self.mu
        Sigma = self.Sigma

        # sample a single point
        z = self.rng.multivariate_normal(mu, Sigma)
        s = target_gaussian_score(z, self.mu_target, self.Sigma_target_inv)

        # intermediate quantities
        eps = Sigma @ s - mu + z
        rhs = s @ Sigma @ s + ((mu - z) @ s) ** 2
        rho = (-1 + np.sqrt(1 + 4 * rhs)) / 2.0

        # delta_mu
        denom = 1 + rho + (mu - z) @ s
        delta_mu = (1 / (1 + rho)) * (np.eye(self.D) - np.outer(mu - z, s) / denom) @ eps

        tilde_mu = mu + delta_mu

        # delta_Sigma
        delta_Sigma = np.outer(mu - z, mu - z) - np.outer(tilde_mu - z, tilde_mu - z)

        # update
        self.mu += delta_mu
        self.Sigma += delta_Sigma

        # jitter for PD
        try:
            np.linalg.cholesky(self.Sigma)
        except np.linalg.LinAlgError:
            self.Sigma += 1e-6 * np.eye(self.D)

    def run(self):
        kl_history = []
        for t in range(self.max_iter):
            self._update()
            kl = kl_gaussian(self.mu, self.Sigma, self.mu_target, self.Sigma_target)
            kl_history.append(kl)
        grad_evals = self.max_iter  # one score evaluation per iteration
        return kl_history, grad_evals