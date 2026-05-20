import numpy as np
from utils import kl_gaussian, inv, det

class ADVI:
    """
    Simple ADVI (gradient descent on KL) for Gaussian target.
    """
    def __init__(self, target_mu, target_Sigma, lr_mu=0.01, lr_Sigma=0.01,
                 max_iter=2000, seed=0):
        self.D = target_mu.shape[0]
        self.mu_target = target_mu
        self.Sigma_target = target_Sigma
        self.Sigma_target_inv = inv(target_Sigma)
        self.lr_mu = lr_mu
        self.lr_Sigma = lr_Sigma
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        # initialise variational parameters
        self.mu = np.zeros_like(target_mu)
        self.Sigma = np.eye(self.D)

    def _grad(self):
        """Gradient of KL(q||p) w.r.t mu and Sigma."""
        inv_p = self.Sigma_target_inv
        diff = self.mu - self.mu_target
        grad_mu = inv_p @ diff

        inv_q = inv(self.Sigma)
        grad_Sigma = 0.5 * (inv_p - inv_q)
        return grad_mu, grad_Sigma

    def run(self):
        kl_history = []
        for t in range(self.max_iter):
            grad_mu, grad_Sigma = self._grad()
            self.mu -= self.lr_mu * grad_mu
            self.Sigma -= self.lr_Sigma * grad_Sigma

            # ensure positive definiteness
            try:
                np.linalg.cholesky(self.Sigma)
            except np.linalg.LinAlgError:
                # add jitter
                self.Sigma += 1e-6 * np.eye(self.D)

            kl = kl_gaussian(self.mu, self.Sigma, self.mu_target, self.Sigma_target)
            kl_history.append(kl)
        grad_evals = 0  # ADVI does not evaluate target scores
        return kl_history, grad_evals