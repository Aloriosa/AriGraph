import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ve_transition_params(T=1.0, sigma_min=0.01, sigma_max=50.0):
    """
    Return functions for variance‑exploding SDE:
        d theta = g(t) dW_t
    g(t) = sigma_min * (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))
    """
    def g(t):
        return sigma_min * (sigma_max / sigma_min) ** t * np.sqrt(2 * np.log(sigma_max / sigma_min))
    def var_t(t):
        return sigma_min ** 2 * (sigma_max / sigma_min) ** (2 * t)
    return g, var_t

def sample_transition(theta0, t, var_t):
    """
    Sample theta_t from p_t(theta_t | theta0) for VE SDE.
    """
    eps = torch.randn_like(theta0)
    return theta0 + torch.sqrt(var_t(t)) * eps

def target_score(theta_t, theta0, t, var_t):
    """
    Score of the transition density p_t(theta_t | theta0).
    For VE SDE: -(theta_t - theta0) / var_t(t)
    """
    return -(theta_t - theta0) / var_t(t)

def ode_integrate(score_net, x, theta0, steps=100, device='cpu'):
    """
    Integrate the probability‑flow ODE backwards from t=1 to t=0.
    Uses simple explicit Euler integration.
    """
    dt = -1.0 / steps  # negative because we go from T=1 to 0
    theta = theta0.clone().to(device)
    for _ in range(steps):
        t = torch.full((theta.size(0), 1), 1.0 + dt, device=device)
        s = score_net(theta, x, t)
        # VE SDE: f=0, g^2 = var_t(t) so velocity = -0.5 * var_t(t) * s
        # We precompute var_t(t) using the same var_t function as above
        # But for simplicity, we approximate var_t(t) as constant (sigma_max^2)
        # because we only need a rough sample for the toy demo.
        # For a full implementation, compute var_t(t) accurately.
        v = -0.5 * var_t(1.0 + dt) * s
        theta = theta + v * dt
    return theta

def kde_hpr(samples, eps=0.05, bandwidth=0.1):
    """
    Estimate the highest probability region (HPR) of the posterior
    using a kernel density estimate.
    Returns a threshold density value such that 1 - eps mass is retained.
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(samples)
    log_dens = kde.score_samples(samples)
    dens = np.exp(log_dens)
    # Compute threshold
    threshold = np.quantile(dens, 1 - eps)
    return threshold, kde

def sample_truncated_prior(prior_sampler, kde, threshold, n_samples, batch=1000):
    """
    Sample from the prior and accept only those whose density under the KDE
    is above the threshold. Rejection sampling.
    """
    accepted = []
    while len(accepted) < n_samples:
        batch_samples = prior_sampler(batch)
        log_dens = kde.score_samples(batch_samples)
        dens = np.exp(log_dens)
        mask = dens >= threshold
        accepted.extend(batch_samples[mask])
    return np.array(accepted[:n_samples])

def c2st_score(samples1, samples2, clf=None, random_state=0):
    """
    Compute the two‑sample classification test (C2ST) accuracy.
    samples1, samples2: (N, d) numpy arrays
    Returns the classification accuracy (higher is worse).
    """
    X = np.concatenate([samples1, samples2], axis=0)
    y = np.concatenate([np.ones(samples1.shape[0]), np.zeros(samples2.shape[0])])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y)
    if clf is None:
        clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc