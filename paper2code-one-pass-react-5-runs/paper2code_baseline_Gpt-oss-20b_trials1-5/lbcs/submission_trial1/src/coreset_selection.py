import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .models import SimpleCNN
from .utils import mask_to_indices, indices_to_mask, set_seed

# ---------- Inner loop training ----------
def train_and_eval(model, train_loader, test_loader, epochs=5, lr=0.01):
    """Train the model on the provided train_loader and evaluate on test_loader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


# ---------- Outer loop (lexicographic bilevel) ----------
def lexicographic_bilevel(
    train_dataset,
    test_loader,
    k: int,
    epsilon: float,
    max_iter: int = 20,
    swap_rate: float = 0.05,
    seed: int = 42,
):
    """
    Simplified implementation of LBCS:
    - Inner loop: train a CNN on the selected subset and evaluate on test set.
    - Outer loop: random search around current best mask using lexicographic
      comparison (loss first, then size).
    """
    set_seed(seed)
    n_samples = len(train_dataset)

    # Initial mask: random k samples
    init_indices = np.random.choice(n_samples, k, replace=False)
    mask = indices_to_mask(init_indices, n_samples)

    best_loss, best_acc = train_and_eval(
        SimpleCNN(),
        DataLoader(Subset(train_dataset, init_indices), batch_size=128),
        test_loader,
    )
    best_mask = copy.deepcopy(mask)
    best_size = k

    print(f"Initial mask size: {k}, test loss: {best_loss:.4f}, acc: {best_acc:.2f}%")

    for it in tqdm(range(1, max_iter + 1), desc="LBCS iterations"):
        # Propose a new mask by swapping a small fraction of indices
        num_swap = max(1, int(swap_rate * k))
        # Indices currently selected
        selected = mask_to_indices(mask)
        # Indices not selected
        not_selected = np.where(mask == 0)[0]

        # Randomly choose to swap
        to_remove = np.random.choice(selected, num_swap, replace=False)
        to_add = np.random.choice(not_selected, num_swap, replace=False)

        new_indices = np.setdiff1d(selected, to_remove)
        new_indices = np.concatenate([new_indices, to_add])
        new_mask = indices_to_mask(new_indices, n_samples)

        # Evaluate new mask
        loss, acc = train_and_eval(
            SimpleCNN(),
            DataLoader(Subset(train_dataset, new_indices), batch_size=128),
            test_loader,
        )
        new_size = new_indices.size

        # Lexicographic comparison:
        # 1) loss within ε of best_loss
        # 2) if loss better or equal, prefer smaller size
        if loss <= best_loss * (1 + epsilon):
            if loss < best_loss or new_size < best_size:
                best_loss, best_acc = loss, acc
                best_mask = copy.deepcopy(new_mask)
                best_size = new_size

    return best_mask, best_size, best_loss, best_acc