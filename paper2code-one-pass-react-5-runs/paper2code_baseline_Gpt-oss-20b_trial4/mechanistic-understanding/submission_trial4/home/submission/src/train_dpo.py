import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

def compute_logprob(model, input_ids, labels):
    """Return the total log‑probability of the sequence."""
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss  # average cross‑entropy
    seq_len = labels.size(1)
    # loss is average, so multiply by seq_len to get sum of negative log‑likelihood
    logp = -loss.item() * seq_len
    return logp

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2-medium"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=200
    )
    beta = 0.1
    batch_size = 4

    # Build a tiny pairwise dataset
    raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:200]")
    pairs = []

    for row in tqdm(raw_ds, desc="Generating pairs"):
        prompt = row["text"].strip()
        if not prompt:
            continue

        # Positive continuation (greedy)
        inp_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            pos_ids = model.generate(
                inp_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Negative continuation – append a toxic word
        neg_prompt = prompt + " sh*t"
        neg_ids = tokenizer.encode(neg_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            neg_ids = model.generate(
                neg_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Store the full sequences (prompt + continuation)
        pos_seq = torch.cat([inp_ids, pos_ids[:, 1:]], dim=1)
        neg_seq = torch.cat([inp_ids, neg_ids[:, 1:]], dim=1)
        pairs.append((pos_seq, neg_seq))

    # Training loop
    for epoch in range(5):
        epoch_loss = 0.0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            loss_batch = 0.0
            for pos_seq, neg_seq in batch:
                # log‑probabilities under current model
                pos_logp = compute_logprob(model, pos_seq, pos_seq)
                neg_logp = compute_logprob(model, neg_seq, neg_seq)

                # log‑probabilities under reference model
                pos_logp_ref = compute_logprob(ref_model, pos_seq, pos_seq)
                neg_logp_ref = compute_logprob(ref_model, neg_seq, neg_seq)

                # DPO loss
                loss = -torch.log(
                    torch.sigmoid(
                        beta
                        * (
                            (pos_logp - pos_logp_ref)
                            - (neg_logp - neg_logp_ref)
                        )
                    )
                )
                loss_batch += loss

            loss_batch = loss_batch / len(batch)
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss_batch.item()

        print(f"Epoch {epoch + 1} / 5 – Avg loss: {epoch_loss / len(pairs):.4f}")

    # Save the fine‑tuned model
    os.makedirs("results", exist_ok=True)
    model.save_pretrained("results/dpo_model")
    tokenizer.save_pretrained("results/dpo_model")
    print("DPO model saved to results/dpo_model")

if __name__ == "__main__":
    main()