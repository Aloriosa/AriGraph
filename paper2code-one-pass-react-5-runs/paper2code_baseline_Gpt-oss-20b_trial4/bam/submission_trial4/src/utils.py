import numpy as np

def log_pdf_gaussian(x, mu, Sigma):
    """
    Log probability density of a multivariate Gaussian (unnormalised).
    """
    D = x.shape[0]
    inv = np.linalg.inv(Sigma)
    diff = x - mu
    return -0.5 * (diff @ inv @ diff) - 0.5 * np.log(np.linalg.det(Sigma)) - 0.5 * D * np.log(2 * np.pi)

def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    Closed‑form KL(q || p) where q = N(mu_q, Sigma_q), p = N(mu_p, Sigma_p).
    """
    inv_p = np.linalg.inv(Sigma_p)
    term1 = np.trace(inv_p @ Sigma_q)
    diff = mu_q - mu_p
    term2 = diff @ inv_p @ diff
    term3 = np.log(np.linalg.det(Sigma_p) / np.linalg.det(Sigma_q))
    D = mu_q.shape[0]
    return 0.5 * (term1 + term2 - D + term3)

def forward_kl(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    KL(p || q) for two Gaussians (forward KL).  This is not analytically simple,
    so we approximate it with a Monte‑Carlo estimate.
    """
    N = 5000
    eps = np.random.randn(N, mu_q.shape[0])
    z = mu_q + eps @ np.linalg.cholesky(Sigma_q).T
    logp = np.array([log_pdf_gaussian(x, mu_p, Sigma_p) for x in z])
    logq = np.array([log_pdf_gaussian(x, mu_q, Sigma_q) for x in z])
    return np.mean(logp - logq)

def score_gaussian(x, mu, Sigma):
    """Gradient of log p(x) for Gaussian target."""
    return np.linalg.inv(Sigma) @ (mu - x)