import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_cifar100_incremental(task_id, classes_per_task, num_tasks,
                             train=True, transform=None):
    """
    Returns a Subset of CIFAR‑100 containing only the classes belonging to
    the specified incremental task.
    """
    dataset = datasets.CIFAR100(root='.', train=train, download=True,
                               transform=transform)
    # Determine class indices for this task
    start_cls = task_id * classes_per_task
    end_cls = start_cls + classes_per_task
    task_classes = list(range(start_cls, end_cls))

    # Map original labels to new labels 0..classes_per_task-1
    idxs = [i for i, (_, lbl) in enumerate(dataset) if lbl in task_classes]
    data = Subset(dataset, idxs)

    # Remap labels
    label_map = {old: new for new, old in enumerate(task_classes)}
    data.dataset.targets = [label_map[dataset.targets[i]] for i in idxs]

    return data


def get_transform(train=True):
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4) if train else transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])