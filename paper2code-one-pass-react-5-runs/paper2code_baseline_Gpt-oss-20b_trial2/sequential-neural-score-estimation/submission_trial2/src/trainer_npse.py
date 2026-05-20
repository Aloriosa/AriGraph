import torch
import time
from tqdm import tqdm
from .npse import NPSE
from .dataset import GaussianLinearDataset, ToyDataset

def train_npse(benchmark: str,
               n_iter: int,
               batch_size: int,
               n_samples: int,
               device: str = "cpu",
               seed: int = 0,
               out_path: str = None):
    """
    Train NPSE on the chosen benchmark.
    """
    if benchmark == "toy":
        dataset = ToyDataset(n_samples)
        dim_theta, dim_x = 1, 1
    elif benchmark == "gaussian_linear":
        dataset = GaussianLinearDataset(n_samples, dim=10)
        dim_theta, dim_x = 10, 10
    else:
        raise ValueError(f"Unknown benchmark {benchmark}")

    model = NPSE(dim_theta, dim_x, device=device, seed=seed)
    best_loss = float("inf")
    best_state = None
    val_split = int(0.15 * n_samples)

    for epoch in tqdm(range(n_iter), desc="Training NPSE"):
        # Shuffle indices
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            theta0, x = dataset.sample(len(batch_idx))
            loss = model.train_step(theta0.to(device), x.to(device), batch_size=len(batch_idx))
        # Validation
        val_loss = 0.0
        n_val = val_split
        for _ in range(5):  # 5 validation batches
            theta0, x = dataset.sample(batch_size)
            with torch.no_grad():
                B = theta0.size(0)
                t = torch.rand(B, device=device)
                theta_t = model.sde.sample(theta0[:B], t)
                target = model.sde.target_score(theta_t, theta0[:B], t)
                pred = model.net(theta_t, x[:B], t)
                val_loss += 0.5 * torch.mean((pred - target) ** 2).item()
        val_loss /= 5
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.net.state_dict()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} – val loss: {val_loss:.4f}")

    if out_path:
        torch.save(best_state, out_path)
    return out_path