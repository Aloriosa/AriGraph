import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_seed, ensure_dir, load_tokenizer, load_model, collate_fn
import yaml

config = yaml.safe_load(open("config.yaml"))
set_seed(config["seed"])
device = config["device"]
probe_cfg = config["probe"]

tokenizer = load_tokenizer(probe_cfg["model_name"])
model = AutoModelForCausalLM.from_pretrained(probe_cfg["model_name"]).to(device)
model.eval()

train_ds = load_dataset("jigsaw-toxic-comment-classification-challenge", split="train")
val_ds = load_dataset("jigsaw-toxic-comment-classification-challenge", split="validation")

# Prepare data: average last hidden state
def encode(example):
    inputs = tokenizer(example["comment_text"], truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device), output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]
    avg = hidden.mean(dim=1)  # [1, hidden]
    label = torch.tensor([example["toxic"]], dtype=torch.float32)
    return {"emb": avg.squeeze(0), "label": label}

train_ds = train_ds.map(encode, remove_columns=train_ds.column_names).remove_columns(["comment_text", "toxic"])
val_ds = val_ds.map(encode, remove_columns=val_ds.column_names).remove_columns(["comment_text", "toxic"])

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=probe_cfg["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=probe_cfg["batch_size"])

linear = nn.Linear(model.config.hidden_size, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(linear.parameters(), lr=probe_cfg["lr"])

ensure_dir(probe_cfg["output_path"])
best_val_loss = float("inf")

for epoch in range(probe_cfg["epochs"]):
    linear.train()
    epoch_loss = 0.0
    for batch in train_loader:
        emb = batch["emb"].to(device)
        label = batch["label"].to(device)
        logits = linear(emb).squeeze(-1)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    # validation
    linear.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            emb = batch["emb"].to(device)
            label = batch["label"].to(device)
            logits = linear(emb).squeeze(-1)
            loss = criterion(logits, label)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{probe_cfg['epochs']} - train loss: {avg_epoch_loss:.4f} - val loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(linear.state_dict(), os.path.join(probe_cfg["output_path"], "probe.pt"))

print("Probe training done. Saved to:", probe_cfg["output_path"])