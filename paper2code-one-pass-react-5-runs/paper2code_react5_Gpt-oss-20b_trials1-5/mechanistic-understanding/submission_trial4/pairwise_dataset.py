import random
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_tokenizer, load_model
import yaml

config = yaml.safe_load(open("config.yaml"))
device = config["device"]
dpo_cfg = config["dpo"]
tokenizer = load_tokenizer(dpo_cfg["base_model"])
base_model = AutoModelForCausalLM.from_pretrained(dpo_cfg["base_model"]).to(device)
base_model.eval()

# Generate pairwise data
def generate_pair(prompt, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Positive: greedy
    pos_out = base_model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pos_text = tokenizer.decode(pos_out[0], skip_special_tokens=True)[len(prompt):].strip()
    # Negative: sampling
    neg_out = base_model.generate(**inputs, max_new_tokens=max_new, do_sample=True, top_k=50, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    neg_text = tokenizer.decode(neg_out[0], skip_special_tokens=True)[len(prompt):].strip()
    return {"prompt": prompt, "positive": pos_text, "negative": neg_text}

# Build dataset from wikitext prompts
prompts = []
for i in range(1000):
    prompts.append(f"Prompt {i}")

data = [generate_pair(p) for p in prompts]
ds = Dataset.from_list(data)
ds.save_to_disk("outputs/pairwise_dataset")
print("Pairwise dataset saved to outputs/pairwise_dataset")