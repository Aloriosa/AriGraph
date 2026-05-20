"""
Train a linear probe on the last‑layer residual stream of a causal LM.

The probe is a single linear layer (hidden_dim → 1) trained with BCE loss.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def train_probe(model, tokenizer, device="cpu",
                epochs=3, batch_size=8,
                train_size=2000, val_size=200):
    """
    Trains a linear probe to predict toxicity from the averaged last‑layer
    hidden state of the language model.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Pre‑trained causal language model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    device : str or torch.device
        Device to run the training on.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    train_size : int
        Number of training examples to sample.
    val_size : int

    Returns
    -------
    probe_weight : torch.Tensor
        Weight vector of the trained probe (shape: [hidden_dim]).
    """
    # Load Jigsaw toxicity dataset (small subset)
    dataset = load_dataset("jigsaw-toxic-comment-classification-challenge", split="train")
    # Jigsaw has columns: 'comment_text', 'toxic' (0/1)
    dataset = dataset.shuffle(seed=42).select(range(train_size + val_size))

    # Tokenize & compute last‑layer hidden states
    def encode_and_get_hidden(example):
        inputs = tokenizer(example["comment_text"], truncation=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
            # Average over tokens -> (hidden_dim)
            avg_hidden = hidden.mean(dim=1).squeeze(0)
        return {"hidden": avg_hidden, "label": torch.tensor(example["toxic"], dtype=torch.float)}

    dataset = dataset.map(encode_and_get_hidden, batched=False,
                          remove_columns=dataset.column_names)

    train_ds = dataset.select(range(train_size))
    val_ds = dataset.select(range(train_size, train_size + val_size))

    train_loader = DataLoader(TensorDataset(
        torch.stack(train_ds["hidden"]),
        torch.stack(train_ds["label"])
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.stack(val_ds["hidden"]),
        torch.stack(val_ds["label"])
    ), batch_size=batch_size)

    hidden_dim = model.config.hidden_size
    probe = torch.nn.Linear(hidden_dim, 1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        for h, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            h, y = h.to(device), y.to(device)
            logits = probe(h).squeeze(-1)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # Validation
        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for h, y in val_loader:
                h, y = h.to(device), y.to(device)
                logits = probe(h).squeeze(-1)
                loss = criterion(logits, y)
                val_loss += loss.item()
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f} | Val loss: {val_loss/len(val_loader):.4f}")

    # Return the probe weight vector (bias is ignored)
    return probe.weight.detach().squeeze(-1)