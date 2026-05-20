import torch
import os
from src.dataset import CIFAR100TaskDataset
from src.models import SEMA
from src.trainer import Trainer
from src.utils import seed_everything

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_tasks = 10
    task_size = 10
    batch_size = 128

    # Build model
    model = SEMA(num_classes=100, pretrained=True).to(device)
    trainer = Trainer(model, device)

    seen_classes = []
    results = []

    for t in range(num_tasks):
        print(f'\n=== Training Task {t+1}/{num_tasks} ===')
        train_ds = CIFAR100TaskDataset(task_idx=t, task_size=task_size, train=True)
        test_ds  = CIFAR100TaskDataset(task_idx=t, task_size=task_size, train=False)

        train_loader = train_ds.get_loader(batch_size=batch_size, shuffle=True)
        test_loader  = test_ds.get_loader(batch_size=batch_size, shuffle=False)

        task_classes = train_ds.get_task_classes()
        seen_classes += task_classes

        trainer.train_task(train_loader, task_classes, epoch=5, threshold=1.0)

        acc = trainer.evaluate(test_loader, seen_classes)
        print(f'Task {t+1} Accuracy on seen classes: {acc*100:.2f}%')
        results.append(acc)

    # Compute metrics
    avg_incremental = sum(results) / len(results)
    final_acc = results[-1]
    with open('results.txt', 'w') as f:
        for i, acc in enumerate(results, 1):
            f.write(f'Task {i} Accuracy: {acc*100:.2f}%\n')
        f.write(f'Average Incremental Accuracy: {avg_incremental*100:.2f}%\n')
        f.write(f'Final Accuracy: {final_acc*100:.2f}%\n')
    print('\nResults written to results.txt')

if __name__ == '__main__':
    main()