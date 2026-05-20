#!/usr/bin/env python3
"""
Demo of assistant‑style prompting with system + user prompts.
Outputs both vanilla and CFG versions for comparison.
"""
import argparse
import torch
from cfg_inference import CFGModel
from tqdm import tqdm


def demo_assistant(
    model_name: str,
    gamma: float,
    system_prompt: str,
    user_prompt: str,
    device: str,
    max_length: int = 150,
):
    """
    Generates two completions: one with γ=1.0, one with γ=gamma.
    Returns the raw outputs.
    """
    baseline = CFGModel(model_name, gamma=1.0, device=device)
    cfg = CFGModel(model_name, gamma=gamma, device=device)

    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
    # Baseline
    out_baseline = baseline.generate(
        prompt=full_prompt,
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        do_sample=True,
        seed=0,
    )
    # CFG
    out_cfg = cfg.generate(
        prompt=full_prompt,
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        do_sample=True,
        seed=0,
    )

    return out_baseline, out_cfg


def main():
    parser = argparse.ArgumentParser(description="Assistant prompt demo with CFG")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model id (default: gpt2)")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="CFG guidance strength (default: 1.5)")
    parser.add_argument("--system", type=str, default="Respond enthusiastically to the following user prompt.",
                        help="System prompt text")
    parser.add_argument("--user", type=str,
                        default="What was the Cambridge Analytica scandal?",
                        help="User prompt text")
    parser.add_argument("--examples", type=int, default=1,
                        help="Number of demo examples (default: 1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _ in tqdm(range(args.examples), desc="Assistant demo"):
        out_baseline, out_cfg = demo_assistant(
            args.model, args.gamma, args.system, args.user, device
        )
        print("\n=== Baseline (γ=1.0) ===")
        print(out_baseline[len(args.system)+len(args.user)+12 :])  # strip prompt
        print("\n=== CFG (γ={}) ===".format(args.gamma))
        print(out_cfg[len(args.system)+len(args.user)+12 :])  # strip prompt


if __name__ == "__main__":
    main()