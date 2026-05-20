import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from .utils import kl_gaussian

class ADVI:
    """
    Automatic Differentiation Variational Inference (ELBO maximisation).
    Uses reparameterisation trick and Adam optimiser.
    """

    def __init__(self, dim, mu0, Sigma0, B=20, lr=0.01, T=200):
        self.D = dim
        self.mu = jnp.array(mu0)
        self.L = jnp.linalg.cholesky(Sigma0)  # lower-triangular L
        self.B = B
        self.lr = lr
        self.T = T
        # Adam state
        self.m_mu = jnp.zeros_like(self.mu)
        self.v_mu = jnp.zeros_like(self.mu)
        self.m_L = jnp.zeros_like(self.L)
        self.v_L = jnp.zeros_like(self.L)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

    def _sample_z(self, rng):
        eps = random.normal(rng, (self.B, self.D))
        z = self.mu + eps @ self.L.T
        return z

    def _elbo(self, rng):
        z = self._sample_z(rng)
        # log p : Gaussian target with zero mean and identity covariance
        logp = -0.5 * jnp.sum((z - 0) ** 2, axis=1)
        # log q : Gaussian with current mu,L
        logq = -0.5 * jnp.sum((z - self.mu) @ jnp.linalg.inv(self.L.T @ self.L) * (z - self.mu), axis=1)
        elbo = jnp.mean(logp - logq)
        return elbo

    @jit
    def _update(self, rng):
        elbo_grad = grad(self._elbo)(rng)
        # Adam update for mu
        self.t += 1
        self.m_mu = self.beta1 * self.m_mu + (1 - self.beta1) * elbo_grad[0]
        self.v_mu = self.beta2 * self.v_mu + (1 - self.beta2) * (elbo_grad[0] ** 2)
        m_hat_mu = self.m_mu / (1 - self.beta1 ** self.t)
        v_hat_mu = self.v_mu / (1 - self.beta2 ** self.t)
        self.mu = self.mu + self.lr * m_hat_mu / (jnp.sqrt(v_hat_mu) + self.eps)
        # Adam update for L (only upper-triangular part)
        # For simplicity we update L elementwise
        self.m_L = self.beta1 * self.m_L + (1 - self.beta1) * elbo_grad[1]
        self.v_L = self.beta2 * self.v_L + (1 - self.beta2) * (elbo_grad[1] ** 2)
        m_hat_L = self.m_L / (1 - self.beta1 ** self.t)
        v_hat_L = self.v_L / (1 - self.beta2 ** self.t)
        self.L = self.L + self.lr * m_hat_L / (jnp.sqrt(v_hat_L) + self.eps)

    def run(self, callback=None):
        rng = random.PRNGKey(0)
        for _ in range(self.T):
            rng, subkey = random.split(rng)
            self._update(subkey)
            if callback is not None:
                # Convert to numpy
                mu_np = np.array(self.mu)
                Sigma_np = np.array(self.L @ self.L.T)
                callback(mu_np, Sigma_np)