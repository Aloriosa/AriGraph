import os
import json
import torch
from tqdm import tqdm

from src.utils import (
    QADataset,
    load_model_and_tokenizer,
    get_candidate_generation,
    compute_embedding,
)
from src.adapter import Adapter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    base_tokenizer, base_model, embedder = load_model_and_tokenizer("distilgpt2")
    base_model.to(device)
    embedder.to(device)
    base_model.eval()

    # Load adapter
    hidden_dim = embedder.config.hidden_size
    adapter = Adapter(hidden_dim).to(device)
    adapter.load_state_dict(torch.load("models/bbox_adapter.pt"))
    adapter.eval()

    # Load test data
    test_ds = QADataset("data/test.tsv", base_tokenizer)
    preds = []
    for sample in tqdm(test_ds, desc="Evaluating"):
        q = sample["question"]
        # Generate candidates
        candidates = get_candidate_generation(
            base_model,
            base_tokenizer,
            prompt=q,
            num_return_sequences=5,
            max_length=30,
            temperature=1.0,
        )
        # Score candidates
        scores = []
        for cand in candidates:
            inp_ids, attn = base_tokenizer(
                f"{q} <|sep|> {cand}",
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            ).values()
            inp_ids = inp_ids.to(device)
            attn = attn.to(device)
            emb = compute_embedding(embedder, inp_ids, attn)
            score = adapter(emb).item()
            scores.append(score)
        # Pick top candidate
        best_idx = scores.index(max(scores))
        best_ans = candidates[best_idx]
        preds.append(best_ans)

    # Write predictions
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")

    # Compute accuracy
    golds = []
    for sample in test_ds:
        golds.append(sample["answer"])
    correct = sum([p.strip().lower() == g.strip().lower() for p, g in zip(preds, golds)])
    accuracy = correct / len(golds)

    metrics = {"accuracy": accuracy}
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy on test set: {accuracy:.4f}")
    print("Predictions written to predictions.txt")
    print("Metrics written to metrics.json")

if __name__ == "__main__":
    main()