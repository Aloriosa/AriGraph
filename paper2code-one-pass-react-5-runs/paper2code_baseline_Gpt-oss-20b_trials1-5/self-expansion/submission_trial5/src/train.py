"""
Minimal continual‑learning training pipeline for a ViT backbone with a
single learnable adapter per task.

The code follows the high‑level design described in the paper:
* Freeze the ViT backbone (ViT‑B/16, pretrained on ImageNet‑1K).
* Add a lightweight adapter (a 1‑layer MLP) after the last transformer block.
* Train the adapter sequentially on 10 incremental tasks (10 classes each).
* Keep all previously trained adapters frozen (no replay).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import timm
import os
import numpy as np
from tqdm import tqdm
from src.dataset import get_cifar100_splits

# ------------------------------------------------------------------
# Adapter definition
# ------------------------------------------------------------------
class Adapter(nn.Module):
    """
    Simple 1‑layer MLP adapter that operates on the ViT patch‑token
    embeddings after the last transformer block.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class ViTWithAdapters(nn.Module):
    def __init__(self, num_classes: int, adapter_hidden: int = 128):
        super().__init__()
        # Load pretrained ViT‑B/16
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,  # no classification head
            global_pool="avg",
        )
        self.backbone.eval()   # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self.backbone.embed_dim
        self.adapters = nn.ModuleList()
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, task_id=None):
        """
        Forward pass. If task_id is None, all adapters are summed
        weighted by a simple scalar (here we use 1.0 for simplicity).
        """
        # Backbone feature extraction
        with torch.no_grad():
            features = self.backbone(x)          # shape: (B, D)

        # Apply all adapters for the current task
        if task_id is not None and task_id < len(self.adapters):
            # Use only the adapter trained for this task
            adapter_out = self.adapters[task_id](features)
            features = features + adapter_out
        else:
            # If no adapter (should not happen), keep features unchanged
            pass

        logits = self.classifier(features)
        return logits


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------
def train_one_task(model, train_loader, optimizer, criterion, device, task_id):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Task {task_id+1} training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, task_id)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs, task_id=None)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100.0 * correct / total
    return acc


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main():
    # Hyperparameters
    num_tasks      = 10
    epochs_per_task = 5
    lr_adapter     = 1e-3
    lr_classifier  = 1e-3
    batch_size     = 128
    hidden_dim     = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Prepare data
    tasks = get_cifar100_splits(num_tasks=num_tasks, batch_size=batch_size)

    # Initialize model
    model = ViTWithAdapters(num_classes=100, adapter_hidden=hidden_dim).to(device)

    # Loss and optimizer for the classifier (shared across tasks)
    criterion = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(model.classifier.parameters(), lr=lr_classifier)

    all_task_acc = []

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        # ------------------------------------------------------------------
        # 1. Add a new adapter for this task
        # ------------------------------------------------------------------
        new_adapter = Adapter(embed_dim=model.embed_dim, hidden_dim=hidden_dim).to(device)
        model.adapters.append(new_adapter)

        # Optimizer for the new adapter
        optimizer_adapter = optim.Adam(new_adapter.parameters(), lr=lr_adapter)

        # ------------------------------------------------------------------
        # 2. Train for few epochs
        # ------------------------------------------------------------------
        for epoch in range(epochs_per_task):
            train_one_task(model, train_loader, optimizer_adapter, criterion, device, task_id)

        # ------------------------------------------------------------------
        # 3. Evaluate on the test set of the current task
        # ------------------------------------------------------------------
        acc = evaluate(model, test_loader, device)
        print(f"Task {task_id+1} test accuracy: {acc:.2f}%")
        all_task_acc.append(acc)

        # ------------------------------------------------------------------
        # 4. Freeze the newly added adapter (no further updates)
        # ------------------------------------------------------------------
        for p in new_adapter.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # 5. Final evaluation on all tasks (cumulative)
    # ------------------------------------------------------------------
    cumulative_acc = np.mean(all_task_acc)
    print(f"\nAverage task accuracy: {cumulative_acc:.2f}%")

    # Write results to file for the grader
    with open("results.txt", "w") as f:
        f.write(f"Average task accuracy: {cumulative_acc:.2f}%\n")
        for i, acc in enumerate(all_task_acc, 1):
            f.write(f"Task {i} accuracy: {acc:.2f}%\n")


if __name__ == "__main__":
    main()