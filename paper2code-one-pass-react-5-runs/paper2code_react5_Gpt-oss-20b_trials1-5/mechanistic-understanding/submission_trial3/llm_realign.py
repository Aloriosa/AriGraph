"""
Optional: Re‑align LLaMA‑2‑7b after DPO by turning on the gates
for the most toxic units.  The script scales the gate weights
to a large positive value (effectively 1 after sigmoid) and
saves the re‑aligned checkpoint as realigned_llama2.
"""

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

# Load DPO fine‑tuned model
dpo_path = "llama2_dpo"
assert os.path.isdir(dpo_path), "DPO checkpoint not found."
model = LlamaForCausalLM.from_pretrained(
    dpo_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

# Load probe and toxic vectors
probe_state = torch.load("probe.pt", map_location="cpu")
W_toxic = probe_state["weight"][1]
W_toxic = W_toxic / W_toxic.norm()

# Load toxic vectors list
toxic_vectors = torch.load("toxic_vectors.pt", map_location="cpu")
top_units = [(b, u) for b, u, _ in toxic_vectors[:7]]  # top 7 units

# Turn gates on (set gate_proj weight for unit to a large positive value)
with torch.no_grad():
    for block_idx, unit_idx in top_units:
        block = model.model.layers[block_idx]
        # gate_proj is a linear layer: weight shape (hidden, hidden)
        # The gate for unit_idx corresponds to the column
        block.gate_proj.weight[:, unit_idx] += 1e3  # effectively 1 after sigmoid

print(f"Turned on gates for {len(top_units)} toxic units.")

# Save re‑aligned checkpoint
save_path = "realigned_llama2"
model.save_pretrained(save_path)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained(save_path)
print(f"Re‑aligned LLaMA‑2‑7b saved to {save_path}")