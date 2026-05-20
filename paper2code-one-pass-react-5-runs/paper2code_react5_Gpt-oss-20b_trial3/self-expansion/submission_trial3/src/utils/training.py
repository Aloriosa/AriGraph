import torch
import time
import os

from .metrics import accuracy, forgetting
from pathlib import Path

def train_sema(model, loaders, output_dir: str = "./checkpoints"):
    """
    Full training over the task sequence. `loaders` is a list of
    tuples (train_loader, test_loader, task_classes).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = model.device
    results = []

    # Keep track of seen classes for evaluation
    seen_classes = []

    for task_idx, (train_loader, test_loader, task_classes) in enumerate(loaders):
        print(f"\n=== Training Task {task_idx+1} ({len(task_classes)} classes) ===")
        seen_classes.extend(task_classes)
        # Update classifier head to new number of classes
        num_seen = len(seen_classes)
        old_head = model.cls_head
        new_head = torch.nn.Linear(model.hidden_dim, num_seen).to(device)
        # Initialize new weights from old head where possible
        with torch.no_grad():
            new_head.weight[:old_head.weight.size(0)] = old_head.weight
            new_head.bias[:old_head.bias.size(0)] = old_head.bias
        model.cls_head = new_head

        start = time.time()
        acc = model.train_on_task(train_loader, test_loader,
                                  epochs_adapter=5,
                                  epochs_rd=20,
                                  lr_adapter=5e-3,
                                  lr_rd=1e-2,
                                  expansion_threshold=1.0)
        duration = time.time() - start
        print(f"Task {task_idx+1} finished in {duration:.2f}s, test acc: {acc*100:.2f}%")

        # Evaluate on all seen tasks
        acc_all = evaluate_all_seen(model, loaders[:task_idx+1])
        results.append((task_idx+1, acc_all))
        # Save checkpoint
        torch.save(model.state_dict(), Path(output_dir) / f"sema_task{task_idx+1}.pt")

    # Final report
    print("\n=== Final Results ===")
    for t, acc in results:
        print(f"After {t} tasks, accuracy on all seen: {acc*100:.2f}%")
    # Write to file
    with open("results.txt", "w") as f:
        for t, acc in results:
            f.write(f"Task {t}: {acc*100:.2f}%\n")
    return results

def evaluate_all_seen(model, loaders):
    """
    Evaluate the model on all tasks seen so far.
    """
    device = model.device
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for train_loader, test_loader, _ in loaders:
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    return correct / total