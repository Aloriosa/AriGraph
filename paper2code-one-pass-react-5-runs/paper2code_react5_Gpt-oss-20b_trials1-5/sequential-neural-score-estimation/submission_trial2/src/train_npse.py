import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.simulator import sample_prior, simulate, sample_reference
from src.diffusion import VESDE
from src.models import ScoreNetwork
from src.utils import estimate_gaussian_params, posterior_density


def sample_prior_truncated(num, mean, cov, epsilon, sigma_prior=1.0):
    """
    Sample from the prior N(0,1) but accept only samples
    with posterior density (approximated by Gaussian(mean, cov))
    greater than the (1-ε)-quantile threshold.
    """
    accepted = []
    # Compute threshold from current posterior samples
    # Use a large pool to estimate density quantile
    pool = torch.randn(10000, 1) * sigma_prior
    pool_dens = posterior_density(pool, mean, cov)
    threshold = torch.quantile(pool_dens, 1 - epsilon).item()

    while len(accepted) < num:
        samp = torch.randn(1, 1) * sigma_prior
        dens = posterior_density(samp, mean, cov).item()
        if dens > threshold:
            accepted.append(samp)
    return torch.cat(accepted, dim=0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Diffusion
    diffusion = VESDE(sigma_min=0.01, sigma_max=10.0)

    # Model
    score_net = ScoreNetwork(theta_dim=1, x_dim=1).to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='none')

    # Dataset (accumulated over rounds)
    all_theta0 = []
    all_x = []

    # Hyper‑parameters
    epsilon = 1e-4  # truncation level
    rounds = args.rounds
    per_round = args.batches // rounds  # number of simulation steps per round

    for r in range(1, rounds + 1):
        print(f"\n=== Round {r} ===")
        # Estimate posterior density from previous round (if r>1)
        if r == 1:
            # Prior: N(0,1)
            mean = torch.tensor([0.0], device=device)
            cov = torch.tensor([[1.0]], device=device)
        else:
            # Use posterior samples from previous round
            prev_samples = prev_posterior_samples
            mean, cov = estimate_gaussian_params(prev_samples)

        # Sample new (θ0, x) pairs from truncated prior
        theta0 = sample_prior_truncated(per_round, mean, cov, epsilon).to(device)
        x = simulate(theta0).to(device)

        all_theta0.append(theta0)
        all_x.append(x)

        # Concatenate all data
        train_theta0 = torch.cat(all_theta0, dim=0)
        train_x = torch.cat(all_x, dim=0)

        # Training loop
        for epoch in range(args.epochs_per_round):
            epoch_loss = 0.0
            num_batches = int((train_theta0.size(0) + args.batch_size - 1) // args.batch_size)
            perm = torch.randperm(train_theta0.size(0))
            for i in range(num_batches):
                idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
                batch_theta0 = train_theta0[idx]
                batch_x = train_x[idx]

                t = torch.rand(batch_theta0.size(0), device=device)
                theta_t, _ = diffusion.forward_transition(batch_theta0, t)
                target = diffusion.score_noise_grad(theta_t, batch_theta0, t)
                pred = score_net(theta_t, batch_x, t)

                lambda_t = 1.0 / (t * (1.0 - t) + 1e-6)
                loss = (criterion(pred, target) * lambda_t.unsqueeze(1)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * idx.size(0)

            avg_loss = epoch_loss / train_theta0.size(0)
            print(f"  Epoch {epoch+1}/{args.epochs_per_round}  Loss: {avg_loss:.6f}")

        # After training, generate many posterior samples to estimate next truncation
        with torch.no_grad():
            prev_posterior_samples = sample_posterior(
                torch.tensor([0.5], device=device).unsqueeze(0),
                num_samples=20000,
                model=score_net,
                diffusion=diffusion,
                device=device
            )

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(score_net.state_dict(), os.path.join(args.out_dir, "score_net.pt"))
    print("\nTraining finished. Model saved to", args.out_dir)

    # Save final posterior samples for evaluation
    torch.save(prev_posterior_samples.cpu(), os.path.join(args.out_dir, "posterior_samples.pt"))
    print("Posterior samples saved for evaluation.")


def sample_posterior(x_obs, num_samples, model, diffusion, device):
    """
    Generate posterior samples conditioned on x_obs using the probability‑flow ODE.
    """
    sigma_T = diffusion.sigma(1.0)
    # Initial samples from reference distribution π
    theta = sample_reference(num_samples, sigma_T).to(device)  # (N,1)

    def ode(t, y):
        y = torch.tensor(y.reshape(-1, 1), device=device, dtype=torch.float32, requires_grad=False)
        t_tensor = torch.full((num_samples, 1), t, device=device)
        with torch.no_grad():
            s = model(y, x_obs.repeat(num_samples, 1), t_tensor.squeeze())
        # Probability‑flow ODE: dθ/dt = -(σ(t)^2 / 2) * score
        return (-(diffusion.sigma_sq(t) / 2.0) * s.cpu().numpy().flatten())

    from scipy.integrate import solve_ivp
    y0 = theta.cpu().numpy().flatten()
    sol = solve_ivp(ode, [1.0, 0.0], y0, method="RK45", rtol=1e-3, atol=1e-6)
    samples = torch.tensor(sol.y[:, -1], device=device).unsqueeze(1)  # (N,1)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TSNPSE on toy Gaussian model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--batches", type=int, default=10000, help="Total simulation steps (across all rounds)")
    parser.add_argument("--epochs-per-round", type=int, default=3, help="Epochs to train per round")
    parser.add_argument("--rounds", type=int, default=5, help="Number of sequential rounds")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    train(args)