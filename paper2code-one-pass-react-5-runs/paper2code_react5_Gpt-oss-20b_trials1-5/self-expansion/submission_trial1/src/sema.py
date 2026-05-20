import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from .model import SEMA


def create_task_splits(dataset, num_tasks=10, seed=0):
    """
    Split the dataset indices into class‑incremental tasks.
    Returns a list of (train_indices, test_indices) per task.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Get unique classes
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    random.shuffle(classes)

    # Map class to indices
    class_to_idx = {c: np.where(targets == c)[0] for c in classes}

    tasks = []
    for i in range(num_tasks):
        task_classes = classes[i * 10 : (i + 1) * 10]
        train_idx = np.hstack([class_to_idx[c] for c in task_classes])
        # For validation, use the test split of those classes
        test_idx = np.hstack([class_to_idx[c] for c in task_classes])
        tasks.append((train_idx, test_idx))
    return tasks


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total