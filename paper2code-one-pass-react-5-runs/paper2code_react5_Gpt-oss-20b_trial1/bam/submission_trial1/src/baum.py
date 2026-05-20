import numpy as np
from .utils import sqrtm_sym, kl_gaussian, kl_gaussian_reverse

class BaM:
    """
    Batch and match algorithm for Gaussian variational inference
    based on the score‑based divergence described in the paper.
    """

    def __init__(self,
                 target_mean,
                 target_cov,
                 dim,
                 init_mean=None,
                 init_cov=None,
                 batch_size=100,
                 lambda_reg=10.0,
                 num_iter=200,
                 seed=42):
        """
        Parameters
        ----------
        target_mean : ndarray, shape (dim,)
            Mean of the target Gaussian p.
        target_cov : ndarray, shape (dim, dim)
            Covariance of the target Gaussian p.
        dim : int
            Dimensionality of the latent space.
        init_mean : ndarray, shape (dim,) or None
            Initial mean of the variational Gaussian q. If None,
            defaults to the origin.
        init_cov : ndarray, shape (dim, dim) or None
            Initial covariance of q. If None, defaults to identity.
        batch_size : int
            Number of samples drawn from q at each iteration.
        lambda_reg : float
            Inverse regularization parameter λ_t (learning‑rate like).
        num_iter : int
            Number of iterations to run.
        seed : int
            Random seed for reproducibility.
        """
        self.dim = dim
        self.target_mean = target_mean
        self.target_cov = target_cov
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.num_iter = num_iter
        self.rng = np.random.default_rng(seed)

        if init_mean is None:
            self.mu = np.zeros(dim)
        else:
            self.mu = init_mean.copy()

        if init_cov is None:
            self.Sigma = np.eye(dim)
        else:
            self.Sigma = init_cov.copy()

        # Pre‑compute the inverse of the target covariance for scores
        self.inv_target_cov = np.linalg.inv(self.target_cov)

    def target_score(self, z):
        """
        Score of the target Gaussian: ∇_z log p(z) = Σ_*^{-1}(μ_* - z)
        """
        return self.inv_target_cov @ (self.target_mean[:, None] - z.T).T

    def solve_covariance(self, U, V):
        """
        Solve the quadratic matrix equation
            Σ U Σ + Σ = V
        for Σ, given symmetric PSD matrices U and V.
        """
        # Compute M = I + 4 U V
        M = np.eye(self.dim) + 4 * U @ V
        sqrtM = sqrtm_sym(M)
        inv_term = np.linalg.inv(np.eye(self.dim) + sqrtM)
        Sigma_next = 2 * V @ inv_term
        return Sigma_next

    def run(self, verbose=True):
        """
        Execute the BaM algorithm.

        Returns
        -------
        history : dict
            Keys: 'mu', 'Sigma', 'kl_forward', 'kl_reverse'.
        """
        history = {
            'mu': [],
            'Sigma': [],
            'kl_forward': [],
            'kl_reverse': []
        }

        for t in range(self.num_iter):
            # 1. Sample batch from current q
            z_samples = self.rng.multivariate_normal(
                mean=self.mu,
                cov=self.Sigma,
                size=self.batch_size
            )

            # 2. Evaluate target scores at samples
            g_samples = self.target_score(z_samples)

            # 3. Compute statistics
            bar_z = np.mean(z_samples, axis=0)
            C = np.cov(z_samples, rowvar=False, bias=True)
            bar_g = np.mean(g_samples, axis=0)
            Gamma = np.cov(g_samples, rowvar=False, bias=True)

            # 4. Compute matrices U and V
            U = self.lambda_reg * Gamma + (self.lambda_reg / (1 + self.lambda_reg)) * np.outer(bar_g, bar_g)
            V = self.Sigma + self.lambda_reg * C + (self.lambda_reg / (1 + self.lambda_reg)) * np.outer(self.mu - bar_z,
                                                                                                   self.mu - bar_z)

            # 5. Update covariance
            Sigma_next = self.solve_covariance(U, V)

            # 6. Update mean
            mu_next = (1 / (1 + self.lambda_reg)) * self.mu + \
                      (self.lambda_reg / (1 + self.lambda_reg)) * (Sigma_next @ bar_g + bar_z)

            # 7. Record history
            history['mu'].append(self.mu.copy())
            history['Sigma'].append(self.Sigma.copy())
            history['kl_forward'].append(
                kl_gaussian_reverse(self.mu, self.Sigma, self.target_mean, self.target_cov)
            )
            history['kl_reverse'].append(
                kl_gaussian(self.mu, self.Sigma, self.target_mean, self.target_cov)
            )

            # 8. Update for next iteration
            self.mu = mu_next
            self.Sigma = Sigma_next

            if verbose and (t + 1) % 20 == 0:
                print(f"Iter {t+1:3d} | KL(fwd)={history['kl_forward'][-1]:.4f} "
                      f"| KL(rev)={history['kl_reverse'][-1]:.4f}")

        return history