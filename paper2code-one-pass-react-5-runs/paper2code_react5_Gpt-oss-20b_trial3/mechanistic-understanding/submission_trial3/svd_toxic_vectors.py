"""
Compute an SVD basis of the toxic vectors extracted above.
The left singular vectors (U) are stored as a list of tensors.
"""

import torch
import numpy as np
from torch.linalg import svd

# Load toxic vectors
top_vectors = torch.load("toxic_vectors.pt", map_location="cpu")
vectors = [v for _, _, v in top_vectors]     # list of tensors (hidden,)
mat = torch.stack(vectors, dim=0).numpy()    # (N, H)

# Compute SVD
U, S, Vt = svd(mat, full_matrices=False)

# Keep the first K basis vectors (e.g., 10)
K = 10
basis = torch.from_numpy(U[:, :K]).float()   # (N, K)

# Save
torch.save(basis, "svd_toxic.pt")
print(f"Saved SVD basis (K={K}) to svd_toxic.pt")