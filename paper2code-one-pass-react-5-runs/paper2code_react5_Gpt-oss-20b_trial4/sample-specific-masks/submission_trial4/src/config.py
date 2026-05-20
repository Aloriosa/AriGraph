"""
Configuration options for the training pipeline.
"""

# ---------------------------------------------
# General training hyper‑parameters
# ---------------------------------------------
EPOCHS = 200
BATCH_SIZE = 256
NUM_WORKERS = 4
BASE_LR = 1e-2          # used for the mask generator and delta
WEIGHT_DECAY = 0.0
LR_SCHEDULE = [100, 145]  # epochs at which to decay LR
LR_GAMMA = 0.1

# ---------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------
DATASET = "cifar10"                # options: cifar10, cifar100, svhn
BACKBONE = "resnet18"              # options: resnet18, resnet50, vit_b32
IMAGE_SIZE = 224                   # target input size for all backbones
NUM_SOURCE_CLASSES = 1000          # ImageNet

# ---------------------------------------------
# Mapping strategy
# ---------------------------------------------
# options: 'random', 'frequent', 'iterative'
MAPPING_STRATEGY = "iterative"

# ---------------------------------------------
# Baseline mask size
# ---------------------------------------------
# options: 'full', 'medium', 'narrow'
BASE_MASK_SIZE = "full"

# ---------------------------------------------
# Other optional settings
# ---------------------------------------------
# Random seed
SEED = 42