"""
Optional: Analyze GLU gating behavior in LLaMA‑2‑7b before and after DPO.
The script will load the base LLaMA‑2‑7b model and an optional DPO
fine‑tuned checkpoint (if available).  It then computes the mean
activation of the gate (sigmoid(W1 x)) for the top toxic units
identified by the probe, and prints the difference.

Note: No large checkpoints are committed to the repo; the script
expects the models to be available locally or downloadable from
HuggingFace.
"""

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from datasets import load_dataset
from tqdm import tqdm

# Hyper‑parameters
MAX_TOXIC = 128 * 128  # same as GPT‑2 extraction
TOP_K = 10            # number of toxic units to examine

# Load probe and toxic vectors (same as GPT‑2)
probe_state = torch.load("probe.pt", map_location="cpu")
W_toxic = probe_state["weight"][1]  # shape (hidden,)
W_toxic = F.normalize(W_toxic, dim=0)

# Load toxic vectors list
toxic_vectors = torch.load("toxic_vectors.pt", map_location="cpu")

# Take top TOP_K units
top_units = [(b, u) for b, u, _ in toxic_vectors[:TOP_K]]
print(f"Analyzing top {TOP_K} toxic units for LLaMA‑2‑7b.")

# Load base LLaMA‑2‑7b
print("Loading base LLaMA‑2‑7b (this may take a few minutes)...")
base = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
base.eval()

# Load DPO fine‑tuned model if available
dpo_path = "llama2_dpo"
if os.path.isdir(dpo_path):
    print(f"Loading DPO fine‑tuned LLaMA‑2‑7b from {dpo_path}...")
    dpo = LlamaForCausalLM.from_pretrained(
        dpo_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    dpo.eval()
else:
    dpo = None
    print("DPO checkpoint not found; analysis will be on base model only.")

# Helper: compute mean gate activation for a set of prompts
def mean_gate_activation(model, units, prompts):
    # register hooks to capture gate outputs
    gate_sums = { (layer_idx, unit_idx): 0.0 for (layer_idx, unit_idx) in units}
    gate_counts = { (layer_idx, unit_idx): 0 for (layer_idx, unit_idx) in units}
    handles = []

    def hook_factory(layer_idx):
        def hook(module, input, output):
            # output shape: (batch, seq_len, hidden)
            gate = torch.sigmoid(output)  # gate activations
            for (lidx, uidx) in units:
                if lidx == layer_idx:
                    # accumulate activation for this unit
                    gate_sum = gate[:, :, uidx].sum().item()
                    gate_counts[(lidx, uidx)] += gate.shape[0] * gate.shape[1]
                    gate_sums[(lidx, uidx)] += gate_sum
        return hook

    # attach hooks
    for idx, layer in enumerate(model.model.layers):
        h = layer.gate_proj.register_forward_hook(hook_factory(idx))
        handles.append(h)

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    for prompt in tqdm(prompts, desc="Running model"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    # remove hooks
    for h in handles:
        h.remove()

    # compute means
    means = {}
    for key in gate_sums:
        if gate_counts[key] > 0:
            means[key] = gate_sums[key] / gate_counts[key]
        else:
            means[key] = 0.0
    return means

# Prepare a small prompt set (e.g., 20 wikitext prompts)
prompts_ds = load_dataset("wikitext",
                         "wikitext-2-raw-v1",
                         split="train[:1%]")
prompts = [ex["text"].strip() for ex in prompts_ds if ex["text"].strip()][:20]

# Compute for base model
print("\nBase model gate activations:")
base_means = mean_gate_activation(base, top_units, prompts)
for key, val in base_means.items():
    print(f"Layer {key[0]} unit {key[1]} mean gate: {val:.4f}")

if dpo:
    print("\nDPO model gate activations:")
    dpo_means = mean_gate_activation(dpo, top_units, prompts)
    for key, val in dpo_means.items():
        print(f"Layer {key[0]} unit {key[1]} mean gate: {val:.4f}")