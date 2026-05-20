import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from dataset import IncrementalCIFAR10
from models import SEMA
import utils
from tqdm import tqdm
import random
import numpy as np

# --------------------------------------------------
# Settings
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TASKS = 5
CLASS_PER_TASK = 2
R = 8
EPOCHS_PER_TASK = 1
EXPAND_THRESHOLD = 1.0  # z‑score threshold
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------------------------------
# Data
# --------------------------------------------------
transform = torch.nn.Sequential(
    torch.nn.Upsample(size=224, mode="bilinear"),
    torch.nn.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
)

train_set = IncrementalCIFAR10(root="./data",
                               train=True,
                               transform=transform,
                               n_tasks=NUM_TASKS,
                               n_classes_per_task=CLASS_PER_TASK,
                               seed=SEED)

test_set = IncrementalCIFAR10(root="./data",
                              train=False,
                              transform=transform,
                              n_tasks=NUM_TASKS,
                              n_classes_per_task=CLASS_PER_TASK,
                              seed=SEED)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = SEMA(backbone_name="vit_base_patch16_224",
             expand_layers=[9, 10, 11],
             r=R).to(DEVICE)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
# Only adapters, routers and RDs are trainable
params = []
for l in model.adapters:
    for adapter in model.adapters[l]:
        params += list(adapter.parameters())
    for router in model.routers[l]:
        params += list(router.parameters())
    for rd in model.rds[l]:
        params += list(rd.parameters())
params += list(model.classifier.parameters())
optimizer = optim.Adam(params, lr=1e-3)

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def compute_reconstruction_error(rd, features):
    """
    features: tensor (B, D)
    returns: tensor of shape (B,)
    """
    recon = rd(features)
    loss = F.mse_loss(recon, features, reduction="none")
    loss = loss.mean(dim=1)  # per sample
    return loss

def check_expansion(model, loader, layer_idx):
    """
    Scan through the loader once to decide whether to add an adapter to the layer.
    Return True if expansion needed.
    """
    layer_str = str(layer_idx)
    if len(model.adapters[layer_str]) == 0:
        # First time: always add an adapter
        return True

    all_errors = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            # Forward to get features at this layer
            h = imgs
            for i, blk in enumerate(model.backbone.blocks):
                h = blk(h)
                if i == layer_idx:
                    break
            # Compute reconstruction error for each RD
            layer_errors = []
            for rd in model.rds[layer_str]:
                err = compute_reconstruction_error(rd, h)
                layer_errors.append(err.mean().item())
            all_errors.append(layer_errors)

    # Compute z‑score per RD
    errors_arr = np.array(all_errors)  # shape (num_samples, num_rds)
    mu = errors_arr.mean(axis=0)
    sigma = errors_arr.std(axis=0) + 1e-6
    z = (errors_arr - mu) / sigma  # shape (N, num_rds)
    # Expansion if all RDs are above threshold for at least one sample
    if (z > EXPAND_THRESHOLD).any(axis=0).all():
        return True
    return False

# --------------------------------------------------
# Training loop
# --------------------------------------------------
task_accuracies = []

for task_id in range(NUM_TASKS):
    print(f"\n=== Training Task {task_id+1}/{NUM_TASKS} ===")
    train_loader = train_set.get_task(task_id)
    test_loader = test_set.get_task(task_id)

    # ----- Expansion decision for each expandable layer -----
    for l in model.expand_layers:
        need_expansion = check_expansion(model, train_loader, l)
        if need_expansion:
            print(f"  Adding adapter to layer {l}")
            model.add_adapter(l)
        else:
            print(f"  No expansion at layer {l}")

    # ----- Training -----
    model.train()
    for epoch in range(EPOCHS_PER_TASK):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # ----- Evaluation on all seen classes so far -----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    task_accuracies.append(acc)
    print(f"Task {task_id+1} accuracy: {acc:.2f}%")

# --------------------------------------------------
# Summary
# --------------------------------------------------
with open("results.txt", "w") as f:
    for i, acc in enumerate(task_accuracies, 1):
        f.write(f"Task {i} accuracy: {acc:.2f}%\n")
    avg_acc = sum(task_accuracies) / len(task_accuracies)
    f.write(f"Average accuracy (last task): {avg_acc:.2f}%\n")

print("\n=== Results ===")
for i, acc in enumerate(task_accuracies, 1):
    print(f"Task {i} accuracy: {acc:.2f}%")
print(f"Average accuracy (last task): {sum(task_accuracies)/len(task_accuracies):.2f}%")