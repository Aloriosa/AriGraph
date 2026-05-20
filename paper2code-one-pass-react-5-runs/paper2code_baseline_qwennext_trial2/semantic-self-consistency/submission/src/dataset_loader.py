import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
import torch

DATASET_DIR = "/home/submission/data"
MODEL_DIR = "/home/submission/models"

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset URLs and metadata
DATASET_CONFIGS = {
    "AQuA-RAT": {
        "name": "aquarat",
        "url": "https://github.com/algoverse/semantic-self-consistency-repro/raw/main/data/AQuA-RAT/test.json",
        "local": os.path.join(DATASET_DIR, "AQuA-RAT/test.json"),
        "type": "math",
        "num_samples": 254
    },
    "SVAMP": {
        "name": "svamp",
        "url": "https://github.com/algoverse/semantic-self-consistency-repro/raw/main/data/SVAMP/test.json",
        "local": os.path.join(DATASET_DIR, "SVAMP/test.json"),
        "type": "math",
        "num_samples": 1000
    },
    "StrategyQA": {
        "name": "strategyqa",
        "url": "https://github.com/algoverse/semantic-self-consistency-repro/raw/main/data/StrategyQA/test.json",
        "local": os.path.join(DATASET_DIR, "StrategyQA/test.json"),
        "type": "commonsense",
        "num_samples": 687
    }
}

def download_dataset(dataset_name):
    """Download dataset if not exists"""
    config = DATASET_CONFIGS[dataset_name]
    local_path = config["local"]
    url = config["url"]
    
    # Create directory
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download if not exists
    if not os.path.exists(local_path):
        print(f"Downloading {dataset_name} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved to {local_path}")
    else:
        print(f"Dataset {dataset_name} already exists at {local_path}")

def load_dataset_from_file(dataset_name):
    """Load dataset from local file"""
    config = DATASET_CONFIGS[dataset_name]
    local_path = config["local"]
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Dataset file not found: {local_path}")
    
    print(f"Loading dataset {dataset_name} from {local_path}")
    
    # Load JSON
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def load_dataset_by_name(dataset_name):
    """Load dataset by name"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return load_dataset_from_file(dataset_name)

def download_all_datasets():
    """Download all datasets"""
    for dataset_name in DATASET_CONFIGS:
        download_dataset(dataset_name)

if __name__ == "__main__":
    # If --download flag provided, download datasets
    import sys
    if "--download" in sys.argv:
        download_all_datasets()
    else:
        print("Dataset loader: Use --download flag to download datasets")