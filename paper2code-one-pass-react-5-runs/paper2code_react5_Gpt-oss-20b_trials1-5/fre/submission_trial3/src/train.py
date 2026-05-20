# -*- coding: utf-8 -*-
"""
Convenience script to train FRE encoder and evaluate.
"""

import os
import argparse

from fre import train_encoder, evaluate_encoder, DEVICE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    parser.add_argument("--encoder-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    encoder, decoder, env = train_encoder(
        args.env, args.encoder_steps, args.batch_size, DEVICE
    )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(args.output_dir, "decoder.pt"))

    eval_results = evaluate_encoder(encoder, env, DEVICE)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write("=== Zero‑Shot Evaluation (MSE) ===\n")
        for name, mse in eval_results.items():
            f.write(f"{name:15s} : {mse:.4f}\n")