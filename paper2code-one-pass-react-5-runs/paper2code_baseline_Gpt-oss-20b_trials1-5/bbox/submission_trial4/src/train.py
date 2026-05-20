import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import set_seed, load_config, read_csv, write_csv
from .blackbox_llm import BlackBoxLLM
from .adapter import Adapter
from transformers import DistilBertTokenizerFast

class QAExampleDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.qas = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        q = self.qas.iloc[idx]
        q_text = q["question"]
        a_text = q["answer"]
        # encode question and answer separately
        q_enc = self.tokenizer(q_text, truncation=True, max_length=self.max_len,
                               return_tensors="pt")
        a_enc = self.tokenizer(a_text, truncation=True, max_length=self.max_len,
                               return_tensors="pt")
        return {
            "question_ids": q_enc["input_ids"].squeeze(0),
            "question_mask": q_enc["attention_mask"].squeeze(0),
            "answer_ids": a_enc["input_ids"].squeeze(0),
            "answer_mask": a_enc["attention_mask"].squeeze(0),
        }

def collate_fn(batch):
    # Pad tensors
    def pad(seq, pad_id=0):
        max_len = max([x.size(0) for x in seq])
        return torch.stack([torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=x.dtype)]) for x in seq])

    question_ids = pad([b["question_ids"] for b in batch])
    question_mask = pad([b["question_mask"] for b in batch])
    answer_ids = pad([b["answer_ids"] for b in batch])
    answer_mask = pad([b["answer_mask"] for b in batch])
    return {
        "question_ids": question_ids,
        "question_mask": question_mask,
        "answer_ids": answer_ids,
        "answer_mask": answer_mask,
    }

def nce_loss(scores: torch.Tensor, pos_indices: torch.Tensor):
    """
    scores: (batch_size, num_candidates)
    pos_indices: index of the positive sample in each row (batch)
    """
    # logits = scores
    logits = scores
    # create target distribution where positive has probability 1
    target = torch.zeros_like(logits)
    target.scatter_(1, pos_indices.unsqueeze(1), 1.0)
    loss = nn.functional.cross_entropy(logits, target.argmax(1))
    return loss

def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = read_csv(os.path.join(cfg["data_dir"], "train.csv"))
    train_ds = QAExampleDataset(train_df, DistilBertTokenizerFast.from_pretrained(cfg["adapter"]["encoder_name"]))
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)

    # Initialize components
    llm = BlackBoxLLM(cfg["llm_name"], device, cfg["generation"])
    adapter = Adapter(cfg["adapter"]["encoder_name"], cfg["adapter"]["dropout"]).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_loss = float("inf")
    for epoch in range(cfg["num_epochs"]):
        adapter.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            q_ids = batch["question_ids"].to(device)
            q_mask = batch["question_mask"].to(device)
            pos_ids = batch["answer_ids"].to(device)
            pos_mask = batch["answer_mask"].to(device)

            batch_size = q_ids.size(0)
            num_candidates = cfg["generation"]["num_candidates"]

            # Generate negative candidates for each example
            negatives = []
            for i in range(batch_size):
                prompt = DistilBertTokenizerFast.from_pretrained(cfg["adapter"]["encoder_name"]).decode(q_ids[i], skip_special_tokens=True)
                candidates = llm.generate(prompt, num_candidates)
                negatives.append(candidates)

            # Prepare tensors for all candidates (positive + negatives)
            all_ids = []
            all_mask = []
            for i in range(batch_size):
                pos = pos_ids[i]
                pos_m = pos_mask[i]
                cand = [pos]
                cand_m = [pos_m]
                for c in negatives[i]:
                    enc = DistilBertTokenizerFast.from_pretrained(cfg["adapter"]["encoder_name"]).encode(c, truncation=True, max_length=64, return_tensors="pt")
                    cand.append(enc.squeeze(0))
                    cand_m.append(torch.ones_like(enc.squeeze(0)))
                all_ids.append(torch.stack(cand))
                all_mask.append(torch.stack(cand_m))

            # Stack to shape (batch, num_candidates, seq_len)
            all_ids = torch.stack(all_ids)  # (B, N, L)
            all_mask = torch.stack(all_mask)

            # Flatten for adapter forward
            B, N, L = all_ids.shape
            ids_flat = all_ids.view(B * N, L).to(device)
            mask_flat = all_mask.view(B * N, L).to(device)

            # Query part is same for all candidates in a row
            q_ids_flat = q_ids.unsqueeze(1).expand(-1, N, -1).contiguous().view(B * N, -1).to(device)
            q_mask_flat = q_mask.unsqueeze(1).expand(-1, N, -1).contiguous().view(B * N, -1).to(device)

            with torch.no_grad():
                # We only need the question; the answer part is already in ids_flat
                pass

            # Score each candidate
            scores = adapter(q_ids_flat, ids_flat, torch.cat([q_mask_flat, mask_flat], dim=1))
            scores = scores.view(B, N)

            # Positive is the first column (index 0)
            loss = nce_loss(scores, torch.zeros(B, dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.4f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(cfg["output_dir"], exist_ok=True)
            torch.save(adapter.state_dict(), os.path.join(cfg["output_dir"], "adapter.pt"))
            print(f"Best adapter saved (loss={best_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)