import torch
import numpy as np

def random_mapping(num_target: int, num_source: int = 1000):
    """Return a random injective mapping of shape (num_target,)"""
    indices = np.random.choice(num_source, num_target, replace=False)
    return torch.tensor(indices, dtype=torch.long)

def compute_mapping(loader, model, num_target: int, num_source: int = 1000, device="cpu"):
    """
    Compute a mapping from target to source labels by counting the most frequent
    source predictions for each target class.  `model` should be a frozen
    ImageNet‑pretrained backbone (e.g. PretrainedBackbone).
    """
    model.eval()
    freq = torch.zeros((num_source, num_target), dtype=torch.int64, device=device)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            for t in range(num_target):
                mask = (labels == t)
                if mask.sum() == 0:
                    continue
                preds_t = preds[mask]
                for s in range(num_source):
                    freq[s, t] += (preds_t == s).sum()

    mapping = []
    used = set()
    for t in range(num_target):
        idx = torch.argmax(freq[:, t]).item()
        while idx in used:
            freq[idx, t] = -1
            idx = torch.argmax(freq[:, t]).item()
        mapping.append(idx)
        used.add(idx)

    return torch.tensor(mapping, dtype=torch.long, device=device)