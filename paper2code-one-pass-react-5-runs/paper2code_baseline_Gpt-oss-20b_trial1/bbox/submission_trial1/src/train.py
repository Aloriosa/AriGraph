import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.utils import (
    set_seed,
    QADataset,
    load_model_and_tokenizer,
    get_candidate_generation,
    compute_embedding,
)
from src.adapter import Adapter

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and models
    base_tokenizer, base_model, embedder = load_model_and_tokenizer("distilgpt2")
    base_model.to(device)
    embedder.to(device)
    base_model.eval()  # we do not fine‑tune the black‑box LM

    # Adapter
    hidden_dim = embedder.config.hidden_size
    adapter = Adapter(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-5)

    # Load data
    train_ds = QADataset("data/train.tsv", base_tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            questions = batch["question"]
            answers = batch["answer"]

            # Generate negative samples
            negatives = []
            for q in questions:
                negs = get_candidate_generation(
                    base_model,
                    base_tokenizer,
                    prompt=q,
                    num_return_sequences=5,
                    max_length=30,
                    temperature=1.0,
                )
                # Remove the gold answer if accidentally generated
                negs = [n for n in negs if n.lower() != answers[questions.index(q)].lower()]
                if len(negs) < 3:
                    # pad with random strings
                    negs += [""] * (3 - len(negs))
                negatives.append(negs[:3])

            # Prepare tensors
            pos_embeds = []
            neg_embeds = []
            for q, a, negs in zip(questions, answers, negatives):
                # Positive
                inp_ids, attn = base_tokenizer(
                    f"{q} <|sep|> {a}",
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                    return_tensors="pt",
                ).values()
                inp_ids = inp_ids.to(device)
                attn = attn.to(device)
                emb = compute_embedding(embedder, inp_ids, attn)
                pos_embeds.append(emb)

                # Negatives
                for neg in negs:
                    inp_ids, attn = base_tokenizer(
                        f"{q} <|sep|> {neg}",
                        truncation=True,
                        max_length=128,
                        padding="max_length",
                        return_tensors="pt",
                    ).values()
                    inp_ids = inp_ids.to(device)
                    attn = attn.to(device)
                    emb = compute_embedding(embedder, inp_ids, attn)
                    neg_embeds.append(emb)

            pos_embeds = torch.cat(pos_embeds, dim=0)  # (batch, hidden)
            neg_embeds = torch.cat(neg_embeds, dim=0)  # (batch*3, hidden)

            # Scores
            pos_scores = adapter(pos_embeds).squeeze(-1)  # (batch,)
            neg_scores = adapter(neg_embeds).squeeze(-1)  # (batch*3,)

            # Ranking‑based NCE loss
            # For each positive, compare against all negatives
            loss = 0.0
            for i in range(pos_scores.size(0)):
                pos = pos_scores[i]
                negs = neg_scores[i * 3 : (i + 1) * 3]
                logits = torch.cat([pos.unsqueeze(0), negs], dim=0)  # (4,)
                loss += F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=device))
            loss /= pos_scores.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    # Save adapter
    os.makedirs("models", exist_ok=True)
    torch.save(adapter.state_dict(), "models/bbox_adapter.pt")
    print("Training finished. Adapter saved to models/bbox_adapter.pt")

if __name__ == "__main__":
    main()