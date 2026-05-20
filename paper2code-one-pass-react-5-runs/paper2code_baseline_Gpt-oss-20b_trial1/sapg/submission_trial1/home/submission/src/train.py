"""
Training script for the toy SAPG implementation.
"""

import argparse
import os
import torch
from gymnasium import make
from env_factory import make_vec_env
from sapg import SAPGTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gym environment id")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--policies", type=int, default=2,
                        help="Number of split policies (>=2)")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Total environment steps to train")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    env = make_vec_env(args.env, args.num_envs, seed=args.seed)

    trainer = SAPGTrainer(
        env,
        num_policies=args.policies,
        policy_kwargs=dict(hidden_dim=64, latent_dim=16,
                           entropy_coef=0.0),
        lambda_off=1.0,
        clip_eps=0.2,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        epochs=4,
        batch_size=64,
        device="cpu"
    )

    trainer.train(total_timesteps=args.steps, steps_per_env=16)

    # Save the trained policy
    os.makedirs("results", exist_ok=True)
    torch.save(trainer.policies[0].state_dict(),
               "results/sapg_policy.pt")
    print("Training finished. Policy saved to results/sapg_policy.pt")

if __name__ == "__main__":
    main()