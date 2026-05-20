import random
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

def get_cifar100_incremental(num_tasks: int = 10, batch_size: int = 32,
                             seed: int = 42):
    """
    Returns a list of (train_loader, test_loader) tuples, one per task.
    Each task contains `num_tasks` classes, i.e. 100 / num_tasks classes.
    """
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)

    classes = list(range(100))
    random.Random(seed).shuffle(classes)

    cls_per_task = 100 // num_tasks
    loaders = []
    for t in range(num_tasks):
        task_classes = classes[t * cls_per_task:(t + 1) * cls_per_task]
        # Filter indices
        train_idx = [i for i, (_, lbl) in enumerate(train_set) if lbl in task_classes]
        test_idx  = [i for i, (_, lbl) in enumerate(test_set)  if lbl in task_classes]
        train_subset = Subset(train_set, train_idx)
        test_subset  = Subset(test_set,  test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_subset,  batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        loaders.append((train_loader, test_loader, task_classes))
    return loaders