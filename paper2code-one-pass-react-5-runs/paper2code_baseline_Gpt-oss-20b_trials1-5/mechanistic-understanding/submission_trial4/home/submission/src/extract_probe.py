import os
import torch
from transformers import GPT2Model, GPT2TokenizerFast
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # Model & tokenizer
    model_name = "gpt2-medium"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load a small portion of the Jigsaw dataset
    dataset = load_dataset("jigsaw-toxic-comment-classification-challenge", split="train[:10%]")

    # Tokenise
    inputs = tokenizer(
        dataset["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass to get last‑layer hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, d)

    # Average over tokens
    avg_hidden = last_hidden.mean(dim=1).cpu().numpy()  # (batch, d)
    labels = np.array(dataset["toxic"])

    # Train a logistic regression probe
    clf = LogisticRegression(max_iter=200, n_jobs=4)
    clf.fit(avg_hidden, labels)

    # Probe vector = weight of the logistic regression
    probe_vector = clf.coef_.squeeze()  # shape (d,)

    # Save
    os.makedirs("results", exist_ok=True)
    np.save("results/probe_vector.npy", probe_vector)
    print("Probe vector saved to results/probe_vector.npy")

if __name__ == "__main__":
    main()