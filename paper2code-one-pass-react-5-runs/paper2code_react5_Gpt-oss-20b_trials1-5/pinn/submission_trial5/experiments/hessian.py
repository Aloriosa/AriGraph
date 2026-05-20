"""
Spectral density estimation via power iteration.

The function `top_k_eigenvalues` returns the k largest eigenvalues of the Hessian
of the loss at the current model parameters.  We use a simple power iteration
with orthogonalization for each eigenvector.
"""

import torch
from torch import Tensor
from typing import List, Tuple

def hvp(loss: Tensor, params: List[Tensor], v: Tensor) -> Tensor:
    """Return Hessian‑vector product Hv."""
    grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_v = torch.autograd.grad(grad, params, grad_outputs=v, retain_graph=True)
    return grad_v


def top_k_eigenvalues(loss: Tensor, params: List[Tensor], k: int = 10, iters: int = 10) -> Tuple[List[float], List[Tensor]]:
    """Return top‑k eigenvalues and eigenvectors (approximate)."""
    eigs = []
    vecs = []

    dim = sum(p.numel() for p in params)
    for _ in range(k):
        # random unit vector
        v = torch.randn(dim, device=params[0].device)
        v = v / v.norm()
        for _ in range(iters):
            # compute Hv
            Hv = hvp(loss, params, v)
            # flatten
            Hv_flat = torch.cat([h.reshape(-1) for h in Hv])
            # power step
            v = Hv_flat / Hv_flat.norm()
        # Rayleigh quotient
        Hv = hvp(loss, params, v)
        Hv_flat = torch.cat([h.reshape(-1) for h in Hv])
        eig = (v @ Hv_flat).item()
        eigs.append(eig)
        vecs.append(v.clone())

        # deflate
        if _ < k - 1:
            for i, vi in enumerate(vecs):
                v = v - (v @ vi) * vi
            if v.norm() > 0:
                v = v / v.norm()
    return eigs, vecs