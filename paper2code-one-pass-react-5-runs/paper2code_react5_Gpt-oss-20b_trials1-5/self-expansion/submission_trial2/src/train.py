import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from tqdm import tqdm

from utils import seed_everything, log_msg, mkdir_if_missing, save_checkpoint
from dataset import get_cifar100_tasks, get_test_loader, get_task_loader
from model import SEMA

# --------------------------- hyper‑parameters ---------------------------
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
WARMUP_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPANSION_ZTHRESH = 1.0
EXPANSION_LAYERS = [8, 9, 10]  # last 3 layers of ViT‑B/16 (0‑based)

# -------------------------------------------------------------------------

def train_one_task(model, dataloader, optimizer, criterion, task_idx, new_adapter_idxs):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(dataloader, desc=f"Task {task_idx} train", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Eval", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return correct / len(dataloader.dataset)

def main(args):
    seed_everything(42)
    mkdir_if_missing(args.output_dir)

    # Load dataset & split into tasks
    dataset, task_indices = get_cifar100_tasks(num_tasks=10, seed=42)
    test_loader = get_test_loader(dataset)

    # Initialize model
    model = SEMA(backbone_name="vit_base_patch16_224",
                 num_classes=100,
                 expansion_layers=EXPANSION_LAYERS,
                 expansion_zthreshold=EXPANSION_ZTHRESH).to(DEVICE)

    # Optimiser for all trainable params
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    all_tasks_acc = []

    for task_id, idxs in enumerate(task_indices):
        log_msg(f"=== Training Task {task_id+1} ===", args.logfile)
        task_loader = get_task_loader(dataset, idxs, batch_size=BATCH_SIZE)

        # --------------------- first epoch: expansion detection ---------------------
        # run one forward pass to populate feature buffer
        model.train()
        with torch.no_grad():
            for xb, _ in task_loader:
                xb = xb.to(DEVICE)
                _ = model(xb)
                break  # only need one batch

        # compute RD stats
        model.compute_rd_stats()

        # decide expansions
        new_adapter_idxs = []
        for l in model.expansion_layers:
            if model.should_expand_layer(l):
                model.expand_layer(l)
                # new adapter index is last in the list
                new_adapter_idxs.append((l, len(model.adapters[l]) - 1))

        log_msg(f"Added adapters at layers: {[l for l,_ in new_adapter_idxs]}", args.logfile)

        # freeze all except new adapters
        model.freeze_except_new(new_adapter_idxs)

        # Re‑build optimizer with new params
        optimizer = optim.AdamW(model.get_new_adapter_params(new_adapter_idxs), lr=LR)

        # --------------------- training loop ---------------------
        for epoch in range(EPOCHS):
            loss = train_one_task(model, task_loader, optimizer, criterion,
                                  task_id+1, new_adapter_idxs)
            log_msg(f"Epoch {epoch+1}/{EPOCHS} loss: {loss:.4f}", args.logfile)

        # --------------------- evaluation ---------------------
        # accumulate seen classes
        seen_indices = []
        for i in range(task_id + 1):
            seen_indices.extend(task_indices[i])
        seen_loader = get_task_loader(dataset, seen_indices, batch_size=BATCH_SIZE)

        acc = evaluate(model, seen_loader)
        all_tasks_acc.append(acc)
        log_msg(f"Task {task_id+1} accuracy on all seen classes: {acc:.4f}", args.logfile)

        # checkpoint
        ckpt_path = os.path.join(args.output_dir, f"task_{task_id+1}.pth")
        save_checkpoint({"model_state": model.state_dict()}, ckpt_path)

    # Final report
    log_msg("\n=== Final Results ===", args.logfile)
    for i, acc in enumerate(all_tasks_acc):
        log_msg(f"After Task {i+1}: {acc:.4f}", args.logfile)
    mean_acc = sum(all_tasks_acc) / len(all_tasks_acc)
    log_msg(f"Average accuracy over all tasks: {mean_acc:.4f}", args.logfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    args.logfile = os.path.join(args.output_dir, "train.log")
    main(args)