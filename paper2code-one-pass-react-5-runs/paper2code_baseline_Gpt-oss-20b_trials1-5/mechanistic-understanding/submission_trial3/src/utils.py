import torch
import random
import numpy as np
from transformers import set_seed
from config import SEED

def set_all_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def average_last_hidden(outputs):
    """
    Average the last hidden state over tokens and batch.
    """
    hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
    return hidden.mean(dim=1)  # (batch, hidden)

def softmax_cross_entropy(logits, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

def compute_toxicity_score(outputs, probe_vector):
    """
    Compute toxicity probability using a linear probe.
    `outputs` is a tensor of shape (batch, hidden)
    `probe_vector` is a 1-d tensor of shape (hidden,)
    """
    logits = outputs @ probe_vector  # (batch,)
    probs = torch.sigmoid(logits)
    return probs.mean().item()