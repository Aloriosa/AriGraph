"""
Train a linear probe on the Jigsaw Toxic Comment Classification dataset.
The probe maps the average last‑layer hidden state of GPT‑2‑medium to a
binary toxicity label and is saved as `probe.pt`.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Model, GPT2TokenizerFast
from tqdm import tqdm

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Hyper‑parameters
BATCH_SIZE = 32
EPOCHS = 3
LR = 5e-5
MAX_LEN = 128

# Load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained("gpt2-medium", output_hidden_states=True)
model.eval()  # we only use it to extract hidden states

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Jigsaw dataset
dataset = load_dataset(
    "jigsaw-toxic-comment-classification-challenge", split="train[:90%]"
)
val_set = load_dataset(
    "jigsaw-toxic-comment-classification-challenge", split="train[90%:]"
)

# Convert labels to binary (1 = toxic)
def to_binary(example):
    example["label"] = 1 if example["toxic"] else 0
    return example

dataset = dataset.map(to_binary)
val_set = val_set.map(to_binary)

# Tokenise
def tokenize(example):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

dataset = dataset.map(tokenize, batched=True)
val_set = val_set.map(tokenize, batched=True)

# DataLoader helper
def collate_fn(batch):
    return {
        k: torch.tensor([b[k] for b in batch], dtype=torch.long)
        for k in ["input_ids", "attention_mask", "label"]
    }

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Probe: linear layer from hidden_size to 2 classes
probe = nn.Linear(model.config.hidden_size, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(probe.parameters(), lr=LR)

best_val = 0.0
for epoch in range(EPOCHS):
    probe.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            out = model(input_ids, attention_mask=attn, output_hidden_states=True)
            hidden = out.hidden_states[-1]          # (B, T, H)
            pooled = hidden.mean(dim=1)             # (B, H)

        logits = probe(pooled)                      # (B, 2)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask=attn, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            pooled = hidden.mean(dim=1)
            logits = probe(pooled)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(probe.state_dict(), "probe.pt")

print(f"Best validation accuracy: {best_val:.4f}")
print("Probe weights saved to probe.pt")