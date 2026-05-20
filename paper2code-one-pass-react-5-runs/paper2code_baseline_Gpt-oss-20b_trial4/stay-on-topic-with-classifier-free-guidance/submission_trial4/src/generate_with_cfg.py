#!/usr/bin/env python3
"""
Classifier‑Free Guidance (CFG) demo for GPT‑2.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def generate_with_cfg(
    prompt: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    gamma: float = 1.5,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_length: int = 50,
    seed: int = 42,
):
    """
    Generate text conditioned on *prompt* using CFG.

    The function performs the following steps for each token:
      1. Compute conditioned logits: model(prompt + generated_so_far)
      2. Compute unconditioned logits: model(generated_so_far)  (empty at start)
      3. Convert to log‑probabilities.
      4. Re‑weight with CFG: logp_unc + γ*(logp_cond – logp_unc)
      5. Apply temperature and top‑p filtering.
      6. Sample next token.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode prompt (no special tokens to keep the prompt clean)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    generated_ids = []

    for _ in range(max_length):
        # Conditioned logits (prompt + generated so far)
        if generated_ids:
            input_ids_cond = torch.cat(
                [prompt_ids, torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)],
                dim=1,
            )
        else:
            input_ids_cond = prompt_ids.unsqueeze(0)

        logits_cond = model(input_ids_cond).logits[:, -1, :]  # shape (1, vocab_size)

        # Unconditioned logits (generated so far only)
        if generated_ids:
            input_ids_unc = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits_unc = model(input_ids_unc).logits[:, -1, :]
        else:
            logits_unc = logits_cond  # no difference at first step

        # Log‑probabilities
        logp_cond = torch.nn.functional.log_softmax(logits_cond, dim=-1)
        logp_unc = torch.nn.functional.log_softmax(logits_unc, dim=-1)

        # CFG re‑weighting
        logp_cfg = logp_unc + gamma * (logp_cond - logp_unc)

        # Temperature scaling
        logp_cfg = logp_cfg / temperature

        # Convert to probabilities
        probs = torch.exp(logp_cfg)

        # Top‑p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find cutoff
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        indices_to_remove = sorted_indices[cutoff]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()

        # Sample next token
        next_token = torch.multinomial(probs, 1).item()
        if next_token == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token)

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text


def main():
    # Load GPT‑2 (small) from Hugging Face
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prompts to test
    prompts = [
        "Translate to French: Hello world.",
        "Write a short story about a dragon.",
    ]

    results = []

    print("\n=== CFG Generation Demo ===\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        out = generate_with_cfg(
            prompt,
            model,
            tokenizer,
            gamma=1.5,
            temperature=1.0,
            top_p=0.9,
            max_length=50,
        )
        print(f"Output: {out}\n")
        results.append((prompt, out))

    # Simple keyword metric
    print("\n=== Keyword Adherence Metric ===")
    for prompt, out in results:
        # Take the last word before punctuation as the keyword
        key = prompt.split()[-1].strip(".,!?")
        count = out.lower().count(key.lower())
        print(f"Keyword '{key}' appears {count} times in output.")


if __name__ == "__main__":
    main()