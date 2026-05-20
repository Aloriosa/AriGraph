import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityClassifier:
    """
    Simple wrapper around the HuggingFace toxicity classifier
    'unitary/toxic-bert'.  Returns a scalar toxicity score
    (higher → more toxic).
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unitary/toxic-bert", use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/toxic-bert"
        ).to(device)

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # We use the probability of the 'toxic' class (label 1)
        probs = torch.softmax(logits, dim=-1)
        toxic_prob = probs[:, 1].item()
        return toxic_prob