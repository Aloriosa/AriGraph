import os
import torch
import yaml
import math
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_tokenizer, load_model, compute_toxicity_score, compute_perplexity, compute_f1, get_realtoxicity_prompts, get_wikitext_prompts

config = yaml.safe_load(open("config.yaml"))
device = config["device"]
eval_cfg = config["eval"]

# Load models
tokenizer = AutoTokenizer.from_pretrained(eval_cfg["toxicity_model"])
tox_clf = None
# If using unitary/toxicbert pipeline
tox_clf = lambda txt: compute_toxicity_score(txt, load_toxicity_classifier(device))

# Load GPT2-medium (unaligned) and DPO model
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
gpt2.eval()
dpo_model = AutoModelForCausalLM.from_pretrained(eval_cfg["toxicity_model"]).to(device)  # placeholder, replace with actual dpo path

# 1. Toxicity on RealToxicityPrompts
prompts = get_realtoxicity_prompts(num=200)
gpt2_scores = []
dpo_scores = []
for p in prompts:
    out_gpt2 = gpt2.generate(tokenizer(p, return_tensors="pt").to(device), max_new_tokens=eval_cfg["max_new_tokens"], do_sample=False)[0]
    txt_gpt2 = tokenizer.decode(out_gpt2, skip_special_tokens=True)[len(p):]
    gpt2_scores.append(compute_toxicity_score(txt_gpt2, tox_clf))

    out_dpo = dpo_model.generate(tokenizer(p, return_tensors="pt").to(device), max_new_tokens=eval_cfg["max_new_tokens"], do_sample=False)[0]
    txt_dpo = tokenizer.decode(out_dpo, skip_special_tokens=True)[len(p):]
    dpo_scores.append(compute_toxicity_score(txt_dpo, tox_clf))

print(f"Avg toxicity GPT2: {sum(gpt2_scores)/len(gpt2_scores):.4f}")
print(f"Avg toxicity DPO: {sum(dpo_scores)/len(dpo_scores):.4f}")

# 2. Perplexity on Wikitext-2
wikitext_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=eval_cfg["wikitext_split"])
ppl = compute_perplexity(gpt2, tokenizer, wikitext_ds)
print(f"Perplexity GPT2: {ppl:.2f}")

# 3. F1 on Wikipedia sentences
wiki_ds = load_dataset("wikipedia", "20220301.en", split="train[:2000]")
prompts = [s["text"][:512] for s in wiki_ds]
references = [s["text"][512:].strip() for s in wiki_ds]
f1 = compute_f1(gpt2, tokenizer, prompts, references)
print(f"F1 GPT2: {f1:.4f}")