import numpy as np
from .utils import solve_quadratic_matrix, log_pdf_gaussian

class BaM:
    """
    Batch and Match (BaM) implementation for Gaussian variational family.
    """

    def __init__(self, dim, mu0, Sigma0, B=20, lambda_reg=1.0, T=200, device=None):
        """
        Parameters
        ----------
        dim : int
            Dimensionality D.
        mu0 : np.ndarray, shape (D,)
            Initial mean.
        Sigma0 : np.ndarray, shape (D,D)
            Initial covariance (positive definite).
        B : int
            Batch size.
        lambda_reg : float
            Inverse regularization parameter λ_t (constant for all iterations).
        T : int
            Number of iterations.
        device : None
            Unused placeholder for compatibility with JAX code.
        """
        self.D = dim
        self.mu = mu0.copy()
        self.Sigma = Sigma0.copy()
        self.B = B
        self.lambda_reg = lambda_reg
        self.T = T

    def step(self, score_func):
        """
        One BaM iteration given a score function s(z) = ∇log p(z).
        """
        # Sample from current Gaussian
        eps = np.random.randn(self.B, self.D)
        z = self.mu + eps @ self.Sigma**0.5 if hasattr(self.Sigma, "__pow__") else self.mu + eps @ np.linalg.cholesky(self.Sigma).T
        # Compute scores
        g = np.array([score_func(zb) for zb in z])  # shape (B,D)

        # Statistics
        bar_z = np.mean(z, axis=0)
        C = np.cov(z, rowvar=False, bias=True)  # shape (D,D)
        bar_g = np.mean(g, axis=0)
        Gamma = np.cov(g, rowvar=False, bias=True)

        # Matrices U and V
        U = self.lambda_reg * Gamma + (self.lambda_reg / (1 + self.lambda_reg)) * np.outer(bar_g, bar_g)
        V = self.Sigma + self.lambda_reg * C + (self.lambda_reg / (1 + self.lambda_reg)) * np.outer(self.mu - bar_z, self.mu - bar_z)

        # Update covariance
        Sigma_new = solve_quadratic_matrix(U, V)

        # Update mean
        mu_new = (1 / (1 + self.lambda_reg)) * self.mu + (self.lambda_reg / (1 + self.lambda_reg)) * (Sigma_new @ bar_g + bar_z)

        self.mu = mu_new
        self.Sigma = Sigma_new

    def run(self, score_func, callback=None):
        """
        Run BaM for T iterations.
        """
        for t in range(self.T):
            self.step(score_func)
            if callback is not None:
                callback(t, self.mu, self.Sigma)