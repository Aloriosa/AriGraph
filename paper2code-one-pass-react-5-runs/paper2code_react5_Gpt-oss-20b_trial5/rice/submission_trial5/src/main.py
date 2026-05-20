#!/usr/bin/env python3
"""
Entry point for the RICE reproduction.

Usage:
    python -m src.main --env Hopper-v3 --timesteps 200000
"""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.rirc.rirc import train_rice  # noqa: E402
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="RICE reproduction script")
    parser.add_argument("--env", type=str, default="Hopper-v3")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mix_prob", type=float, default=0.25)
    parser.add_argument("--rnd_coef", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    train_rice(
        env_id=args.env,
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        mix_prob=args.mix_prob,
        rnd_coef=args.rnd_coef,
        alpha=args.alpha,
        log_dir=args.log_dir,
    )

if __name__ == "__main__":
    main()