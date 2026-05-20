import os
import torch

def load_checkpoint(model, path):
    if os.path.isfile(path):
        print(f"Loading checkpoint from {path}")
        state_dict = torch.load(path, map_location=model.device)
        model.load_state_dict(state_dict)
    else:
        print(f"No checkpoint found at {path}")