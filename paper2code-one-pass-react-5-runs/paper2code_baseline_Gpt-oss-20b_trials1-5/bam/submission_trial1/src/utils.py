"""Utility helpers for matrix operations."""
import numpy as np

def inv_sqrtm(A: np.ndarray) -> np.ndarray:
    """Inverse of the matrix square root of A (A must be symmetric positive definite)."""
    w, V = np.linalg.eigh(A)
    assert np.all(w > 0), "Matrix is not positive definite."
    return (V * (1 / np.sqrt(w))) @ V.T

def sqrtm(A: np.ndarray) -> np.ndarray:
    """Matrix square root of A (symmetric positive definite)."""
    w, V = np.linalg.eigh(A)
    assert np.all(w > 0), "Matrix is not positive definite."
    return (V * np.sqrt(w)) @ V.T