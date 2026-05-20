"""
Placeholder AI‑feedback module.
In the original paper, an advanced LLM (e.g., GPT‑4) is used to rank candidates.
Here we provide a lightweight scorer based on GPT‑2 log‑probabilities
to emulate that behaviour in an open‑source setting.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

class GPT2Scorer:
    """
    Computes a simple average token log‑probability for a list of texts
    given a prompt. The higher the score, the more likely the text under GPT‑2.
    """
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()

    def score(self, texts, prompt):
        scores = []
        for text in texts:
            full = prompt + text
            inputs = self.tokenizer(full, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            # higher negative loss (i.e., lower loss) => better
            scores.append(-loss.item() * input_ids.size(1))
        return scores