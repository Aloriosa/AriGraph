"""
Main training script for SEMA on CIFAR‑10 synthetic class‑incremental tasks.
"""

import torch
import numpy as np
from tqdm import tqdm
from utils import get_cifar10_tasks, evaluate
from sema import SEMA

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load synthetic tasks
    train_loaders, test_loader = get_cifar10_tasks(num_tasks=5,
                                                  classes_per_task=2,
                                                  batch_size=256,
                                                  seed=42)

    model = SEMA(num_expandable_layers=3,
                 adapter_dim=64,
                 hidden_dim=768,
                 num_tasks=5).to(device)

    # Tracking accuracy per task
    task_acc = []

    for task_id, loader in enumerate(train_loaders):
        print(f'\n=== Training Task {task_id+1} ===')

        # Scan for expansion (first epoch only)
        expansions = model.scan_task(loader, device, threshold=1.0, epochs=1)
        print('Expansion decisions:', expansions)

        # Expand layers where needed
        for layer_key, expand in expansions.items():
            if expand:
                print(f'  Expanding layer {layer_key}')
                model.expand_layer(layer_key, device)

        # Train on the task
        model.train_task(loader, device, lr=1e-4, epochs=5)

        # Evaluate on test set
        acc = evaluate(model, test_loader, device)
        print(f'Task {task_id+1} accuracy: {acc * 100:.2f}%')
        task_acc.append(acc)

    avg_acc = np.mean(task_acc)
    print(f'\nFinal average accuracy: {avg_acc * 100:.2f}%')

    # Save results
    results = {
        'task_accuracy': [float(a) for a in task_acc],
        'average_accuracy': float(avg_acc),
    }
    import json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print('Results saved to results.json')

if __name__ == '__main__':
    main()