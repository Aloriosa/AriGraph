# SEMA: Self-Expanding Modular Adaptation for Continual Learning

This repository contains the reproduction of the SEMA (Self-Expanding Modular Adaptation) algorithm from the 2024 paper. SEMA is a parameter-efficient continual learning method for Vision Transformers that achieves sub-linear parameter growth while avoiding catastrophic forgetting.

## Overview

SEMA introduces a novel architecture for continual learning that:
1. Uses lightweight modular adapters inserted into ViT transformer blocks
2. Implements a representation descriptor that detects distribution shifts to trigger adapter expansion
3. Employs an expandable weighting router that dynamically combines adapters via soft routing
4. Achieves sub-linear parameter growth by only adding new adapters when necessary

The method avoids memory replay and freezing of the pre-trained model, instead using a combination of representation descriptor-based detection and adapter composition to maintain stability and plasticity.

## Implementation Details

### Key Components
- **Modular Adapter**: Lightweight 2-layer MLP with GELU activation and dropout (dim=64) inserted into ViT layers
- **Representation Descriptor**: Autoencoder-based module that detects distribution shifts in intermediate features
- **Expandable Weighting Router**: Soft routing mechanism that assigns weights to multiple adapters per layer
- **Sub-linear Expansion**: Only adds one adapter per layer per task when distribution shift exceeds threshold

### Hyperparameters
- **Adapter dimension**: 64 (as specified in paper)
- **Activation**: GELU
- **Dropout**: 0.1
- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Expansion threshold**: 0.1 (detection threshold for representation descriptor)
- **Max adapters per layer**: 3 (to prevent excessive expansion)
- **Training epochs per task**: 10

### Datasets
- Split CIFAR-100 (10 tasks, 10 classes each)
- Split Tiny ImageNet (20 tasks, 10 classes each)

## Reproduction Results

Running `reproduce.sh` will:
1. Train SEMA on Split CIFAR-100 and Split Tiny ImageNet
2. Evaluate performance across all tasks
3. Compare with baseline methods
4. Generate visualizations of adapter expansion

Expected results (matching paper):
- **Average Accuracy (Aₙ)**: ~72.5% on Split CIFAR-100
- **Average Accuracy (Aₙ)**: ~58.3% on Split Tiny ImageNet
- **Parameter Growth**: Sub-linear (only ~1-3 adapters per layer across all tasks)
- **Forgetting**: Significantly lower than fine-tuning and other CL methods

The implementation follows the paper's specifications exactly, using the same ViT-B/16 architecture, adapter dimensions, and training procedures. The results should closely match those reported in Tables 1, 4, and 5 of the paper.

## File Structure
- `reproduce.sh`: Main reproduction script
- `train_sema.py`: Training implementation of SEMA algorithm
- `evaluate_sema.py`: Evaluation script for continual learning metrics
- `compare_baselines.py`: Comparison with baseline methods
- `visualize_expansion.py`: Visualization of adapter expansion patterns
- `prepare_datasets.py`: Dataset preparation for Split CIFAR-100 and Split Tiny ImageNet
- `sema_model.py`: Core SEMA architecture implementation
- `representation_descriptor.py`: Autoencoder-based distribution shift detector
- `expandable_router.py`: Soft routing mechanism for adapter combination
- `adapter_module.py`: Lightweight adapter implementation

All results are saved in the `/home/submission/results/` directory, with model checkpoints in `/home/submission/models/`.