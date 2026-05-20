"""
Configuration constants used throughout the repository.
"""

# Random seeds for reproducibility
SEED = 42

# Dataset splits (small for speed)
JIGSAW_SPLIT = ("train[:10%]", "validation[:10%]")  # 10% of each split
WIKITEXT_SPLIT = ("train[:0.01]", "validation[:0.01]")  # 1% for quick eval
REALTOXICITY_PROMPTS = 50  # number of prompts to evaluate

# Hyperparameters
PROBE_HIDDEN_DIM = 768
PROBE_EPOCHS = 3
PROBE_BATCH_SIZE = 32

DPO_EPOCHS = 2
DPO_BATCH_SIZE = 4
DPO_BETA = 0.1
DPO_LR = 1e-6
DPO_GRAD_CLIP = 5.0

# Path constants
BASE_MODEL = "gpt2-medium"