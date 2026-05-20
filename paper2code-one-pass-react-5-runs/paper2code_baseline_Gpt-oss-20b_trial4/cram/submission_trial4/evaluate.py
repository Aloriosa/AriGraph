#!/usr/bin/env python3
"""
Evaluate the reconstruction quality of the learned memory vectors.
"""

import argparse
import os
import torch
import transformers

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory vectors")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory containing mem.pt and text.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load memory vectors & original text
    mem_params = torch.load(
        os.path.join(args.output_dir, "mem.pt"), map_location=device
    )
    data = torch.load(
        os.path.join(args.output_dir, "text.pt"), map_location="cpu"
    )
    token_ids = data["token_ids"].to(device)
    text = data["text"]

    # Cross‑entropy before training (no memory)
    with torch.no_grad():
        outputs_no_mem = model(
            input_ids=token_ids.unsqueeze(0),
            labels=token_ids.unsqueeze(0),
        )
        ce_before = outputs_no_mem.loss.item()

    # Cross‑entropy after training (with memory)
    token_embeds = model.transformer.wte(token_ids)
    inputs_embeds = torch.cat([mem_params, token_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones(1, inputs_embeds.size(1), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs_mem = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=token_ids,
        )
        ce_after = outputs_mem.loss.item()

    # Generate text from memory
    max_new_tokens = token_ids.size(0)
    # GPT‑2 needs a pad_token_id for generate (use EOS)
    pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    generated_ids = model.generate(
        inputs_embeds=mem_params.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        do_sample=False,
    )
    generated_ids = generated_ids[0][:max_new_tokens]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute token accuracy
    correct = (generated_ids == token_ids).sum().item()
    acc = correct / token_ids.size(0)

    # Print results
    print("\n=== Evaluation ===")
    print(f"Original text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"Generated text: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
    print(f"Token accuracy: {acc * 100:.2f}%")
    print(f"Cross‑entropy before (no mem): {ce_before:.4f}")
    print(f"Cross‑entropy after  (with mem): {ce_after:.4f}")
    print(f"Cross‑entropy reduction: {ce_before - ce_after:.4f}")

if __name__ == "__main__":
    main()