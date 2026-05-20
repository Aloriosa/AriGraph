import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import json
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_msg(msg, logfile):
    print(msg)
    if logfile:
        with open(logfile, "a") as f:
            f.write(msg + "\n")

def save_checkpoint(state, ckpt_path):
    torch.save(state, ckpt_path)

def load_checkpoint(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu")

def compute_zscore(errors, mu, sigma):
    return (errors - mu) / (sigma + 1e-8)

def compute_stats(errors):
    return errors.mean(), errors.std()

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)