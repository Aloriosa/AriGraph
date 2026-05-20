#!/usr/bin/env python
import argparse
from src.trainer_npse import train_npse

def main():
    parser = argparse.ArgumentParser(description="Train NPSE on a benchmark")
    parser.add_argument("--benchmark", type=str, default="toy",
                        help="Benchmark to train on: 'toy' or 'gaussian_linear'")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Total number of simulations")
    args = parser.parse_args()

    out_path = f"npse_{args.benchmark}_{args.n_samples}.pt"
    train_npse(args.benchmark, args.max_iter, args.batch_size,
               args.n_samples, device="cpu", seed=args.seed, out_path=out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    main()