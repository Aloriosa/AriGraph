"""
Utility script to load a previously trained memory vector and evaluate
on a new text. Not used in the reproduction script but kept for
reference.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_tokenizer_and_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MEM]"]})
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return tokenizer, model


def evaluate_memory(
    tokenizer, model, mem_token_id, memory_vec, text, device, max_len=None
):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if max_len is None:
        max_len = len(tokens) - 1

    # Greedy decode
    generated_ids = []
    mem_emb = memory_vec.to(device)
    for _ in range(max_len):
        prefix_ids = torch.tensor([mem_token_id] + generated_ids, dtype=torch.long).to(
            device
        )
        embeddings = model.get_input_embeddings()(prefix_ids)
        embeddings[0] = mem_emb
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1).item()
        if next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: python evaluate.py <model_name> <mem_vec_path> <sample_text> <output_json>"
        )
        sys.exit(1)

    model_name, mem_vec_path, sample_text, output_json = sys.argv[1:5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_tokenizer_and_model(model_name, device)
    mem_token_id = tokenizer.convert_tokens_to_ids("[MEM]")

    memory_vec = torch.load(mem_vec_path, map_location=device)
    output = evaluate_memory(
        tokenizer, model, mem_token_id, memory_vec, sample_text, device
    )
    with open(output_json, "w") as f:
        json.dump({"generated_text": output}, f)
    print(f"Generated: {output}")