#!/usr/bin/env python3
"""
Wrapper that runs compress.py on a directory of texts for several numbers of [mem] vectors.
It writes a separate metrics.json for each vector count.
"""

import argparse
import subprocess
from pathlib import Path

def run_compress(text_dir: Path, model_name: str, mem_vectors: int, steps: int, lr: float, threshold: float):
    cmd = [
        "python",
        "compress.py",
        "--model_name",
        model_name,
        "--text_dir",
        str(text_dir),
        "--output_dir",
        f"output/mem_{mem_vectors}",
        "--mem_vectors",
        str(mem_vectors),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--threshold",
        str(threshold),
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run compression experiment for multiple [mem] vector counts."
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        required=True,
        help="Directory containing text files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilgpt2",
        help="HuggingFace model id.",
    )
    parser.add_argument(
        "--mem_counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="List of [mem] vector counts to try.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of optimisation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Accuracy threshold for decoding capacity.",
    )
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory {text_dir} does not exist")

    for k in args.mem_counts:
        print(f"\n=== Running experiment with {k} [mem] vectors ===")
        run_compress(text_dir, args.model_name, k, args.steps, args.lr, args.threshold)

    print("\nAll experiments finished. Results are in the 'output/' directory.")

if __name__ == "__main__":
    main()