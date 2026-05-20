import json
import os
import torch
import numpy as np

def save_results(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        return checkpoint.get("step", 0)
    return 0

def save_checkpoint(path, step, model, optimizer):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }, path)