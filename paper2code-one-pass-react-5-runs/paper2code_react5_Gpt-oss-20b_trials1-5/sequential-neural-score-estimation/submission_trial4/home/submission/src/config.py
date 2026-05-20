import yaml
import os

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg