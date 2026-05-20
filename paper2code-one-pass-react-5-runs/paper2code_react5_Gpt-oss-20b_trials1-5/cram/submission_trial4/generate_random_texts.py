#!/usr/bin/env python3
"""
Generate a set of random‑word text files for compression experiments.
Each file contains a random sequence of tokens of a specified length.
"""

import argparse
import random
from pathlib import Path

def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def generate_random_texts(
    vocab, lengths, num_files, output_dir, seed=42
):
    random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, length in enumerate(lengths, 1):
        for j in range(1, num_files + 1):
            tokens = random.choices(vocab, k=length)
            text = " ".join(tokens) + "."
            fname = f"{i:03d}_{j:02d}.txt"
            Path(output_dir, fname).write_text(text, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(
        description="Create random‑word text files for compression experiments."
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="random_word_list.txt",
        help="Path to a file containing one word per line.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="random_texts",
        help="Directory to write the generated text files.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Sequence lengths (in tokens) to generate.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=3,
        help="Number of files per length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_file)
    generate_random_texts(
        vocab=vocab,
        lengths=args.lengths,
        num_files=args.num_files,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Generated {len(args.lengths)*args.num_files} files in {args.output_dir}")

if __name__ == "__main__":
    main()