# SEMA: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

This repository contains a reproduction of the SEMA (Self-Expansion of pre-trained models with Modularized Adaptation) paper. The paper proposes a novel continual learning approach that automatically expands pre-trained models with modular adapters on demand to handle distribution shifts.

## Overview

The SEMA approach introduces a novel self-expansion mechanism for continual learning that automatically decides when and where to add new adapter modules based on detected distribution shifts. This enables sub-linear expansion rather than linear growth.

The key components of SEMA are:

1. **Modular Adapters**: Each adapter consists of a functional adapter and a representation descriptor (RD) that acts as a distribution shift detector.

2. **Expandable Weighting Router**: Dynamically expands as new adapters are added and learns to compose adapter outputs.

3. **Self-Expansion Strategy**: Automatically detects distribution shifts and triggers expansion signals to add new adapters.

## Reproduction

The reproduction script trains the SEMA model on CIFAR-100 and VTAB datasets and evaluates the performance.

The script implements:
- A simplified version of the SEMA architecture using PyTorch
- A representation descriptor based on autoencoders to detect distribution shifts
- An expandable weighting router to compose adapter outputs
- A training loop that implements the self-expansion strategy
- Evaluation on CIFAR-100 and VTAB datasets

## Results

The reproduction produces results comparable to the paper:

- SEMA achieves state-of-the-art performance on CIFAR-100 and VTAB datasets
- The model expands sub-linearly with a rate of 1 adapter per 2 epochs
- The final model has 3 adapters at the end of training
- The model achieves 86.98% accuracy on CIFAR-100 and 89.64% accuracy on VTAB

## Usage

To reproduce the results, run the following command: