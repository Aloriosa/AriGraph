import torch
import numpy as np
import os
import json
from .config import load_config
from .training import sequential_training
from .sampling import sample_posterior
from .utils import log_posterior_gaussian

def main():
    cfg = load_config("config.yaml")
    os.makedirs("outputs", exist_ok=True)

    # Train model
    model, x_obs = sequential_training(cfg)

    # Save model
    torch.save(model.state_dict(), "outputs/model.pt")

    # Sample from posterior
    samples = sample_posterior(model, x_obs, cfg, device=cfg["device"],
                               num_samples=cfg["num_samples"],
                               sample_steps=cfg["sample_steps"])
    np.save("outputs/samples.npy", samples.cpu().numpy())

    # Compute posterior mean and variance
    mean = samples.mean(axis=0)
    var = samples.var(axis=0, ddof=1)

    # Analytical posterior for Gaussian‑linear
    prior_var = cfg["prior_std"] ** 2
    sim_var   = cfg["simulator_std"] ** 2
    post_var  = 1.0 / (1.0 / prior_var + 1.0 / sim_var)
    # Prior mean defaults to 0; extendable if prior_mean is specified
    prior_mean = cfg.get("prior_mean", 0.0)
    # Correct posterior mean: μ_post = σ_post² * (x / σ_sim² + μ_prior / σ_prior²)
    post_mean = post_var * (cfg["x_observation"] / sim_var + prior_mean / prior_var)

    # Errors
    mean_err = (mean.item() - post_mean) ** 2
    var_err  = (var.item()  - post_var)  ** 2

    # Save metrics
    with open("outputs/posterior_mean.txt", "w") as f:
        f.write(f"Posterior mean estimate: {mean.item():.6f}\n")
        f.write(f"True posterior mean: {post_mean:.6f}\n")
        f.write(f"Squared error: {mean_err:.6e}\n")
        f.write(f"Posterior variance estimate: {var.item():.6f}\n")
        f.write(f"True posterior variance: {post_var:.6f}\n")
        f.write(f"Variance squared error: {var_err:.6e}\n")

    # Simple report
    report = {
        "posterior_mean": mean.item(),
        "posterior_variance": var.item(),
        "true_posterior_mean": post_mean,
        "true_posterior_variance": post_var,
        "mean_squared_error": mean_err,
        "variance_squared_error": var_err
    }
    with open("outputs/report.txt", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Reproduction finished ===")
    print(f"Posterior mean estimate: {mean.item():.6f}")
    print(f"True posterior mean: {post_mean:.6f}")
    print(f"Mean squared error: {mean_err:.6e}")
    print(f"Posterior variance estimate: {var.item():.6f}")
    print(f"True posterior variance: {post_var:.6f}")
    print(f"Variance squared error: {var_err:.6e}")

if __name__ == "__main__":
    main()