"""
benchmark_lambada.py

Minimal benchmark on the LAMBADA dataset that compares vanilla
generation vs. CFG (γ=1.5).  Only a small subset of the validation split
is used (first 200 examples) to keep runtime reasonable.

Usage
-----
>>> python benchmark_lambada.py
"""

import torch
from datasets import load_dataset
from cfg_inference import CfgGenerator, load_gpt2

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "gpt2-medium"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1
GAMMA = 1.5
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def tokenize_batch(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")


def compute_accuracy(preds, targets):
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    # Load model and tokenizer
    data = load_gpt2(MODEL_NAME)
    model, tokenizer = data["model"], data["tokenizer"]
    model.to(DEVICE)
    model.eval()

    # Load LAMBADA validation split
    dataset = load_dataset("lambada", split="validation")
    # Use only first 200 examples to keep runtime short
    dataset = dataset.select(range(200))

    # Prepare inputs
    contexts = []
    targets = []
    for ex in dataset:
        text = ex["text"]
        # LAMBADA: the last word is the target
        tokens = tokenizer.tokenize(text)
        target_token = tokens[-1]
        context = tokenizer.convert_tokens_to_string(tokens[:-1])
        contexts.append(context)
        targets.append(tokenizer.convert_tokens_to_ids([target_token])[0])

    # Vanilla generation
    vanilla_gen = CfgGenerator(
        model, tokenizer, gamma=1.0, device=DEVICE
    )
    cfg_gen = CfgGenerator(
        model, tokenizer, gamma=GAMMA, device=DEVICE
    )

    vanilla_preds = []
    cfg_preds = []

    for i in range(0, len(contexts), BATCH_SIZE):
        batch_ctx = contexts[i : i + BATCH_SIZE]
        batch_targets = targets[i : i + BATCH_SIZE]

        # Generate one token for each context
        for ctx, tgt in zip(batch_ctx, batch_targets):
            out_vanilla = vanilla_gen.generate(
                ctx,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                return_full_text=False,
            )
            out_cfg = cfg_gen.generate(
                ctx,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                return_full_text=False,
            )

            # Token ids of first token of the output
            out_vanilla_id = tokenizer.encode(out_vanilla, add_special_tokens=False)[0]
            out_cfg_id = tokenizer.encode(out_cfg, add_special_tokens=False)[0]

            vanilla_preds.append(out_vanilla_id)
            cfg_preds.append(out_cfg_id)

    # Compute accuracies
    vanilla_acc = compute_accuracy(vanilla_preds, targets)
    cfg_acc = compute_accuracy(cfg_preds, targets)

    print("\n=== LAMBADA (first 200 examples) ===")
    print(f"Vanilla γ=1.0 accuracy: {vanilla_acc * 100:.2f}%")
    print(f"CFG   γ={GAMMA} accuracy: {cfg_acc * 100:.2f}%")
    print(f"Δ improvement: { (cfg_acc - vanilla_acc) * 100:.2f}%\n")


if __name__ == "__main__":
    main()