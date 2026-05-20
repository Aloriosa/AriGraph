import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    """Simple accuracy function."""
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")