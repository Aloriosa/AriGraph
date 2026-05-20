import numpy as np
from utils import kl_gaussian

def advi_step(mu, Sigma, mu_target, Sigma_target, lr):
    """
    One ADVI iteration: gradient descent on closed‑form KL(q||p).
    """
    # gradients
    inv_target = np.linalg.inv(Sigma_target)
    g_mu = inv_target @ (mu - mu_target)          # dKL/dmu
    g_Sigma = 0.5 * (inv_target - np.linalg.inv(Sigma))  # dKL/dSigma

    # simple gradient descent update
    mu_new = mu - lr * g_mu
    # For Sigma, we update in terms of its Cholesky factor L: Sigma = L L^T.
    # Here we perform a naive update on Sigma itself and re‑symmetrize.
    Sigma_new = Sigma - lr * g_Sigma
    # enforce symmetry
    Sigma_new = (Sigma_new + Sigma_new.T) / 2
    # ensure positive definiteness by projection
    eigvals, eigvecs = np.linalg.eigh(Sigma_new)
    eigvals = np.clip(eigvals, 1e-6, None)
    Sigma_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return mu_new, Sigma_new