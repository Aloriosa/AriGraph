# SEMA: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

This repository contains the reproduction of the SEMA (Self-Expansion of pre-trained models with Modularized Adaptation) algorithm from the paper "Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning".

## Overview

The SEMA algorithm implements a novel approach for continual learning that addresses the catastrophic forgetting problem in pre-trained models by automatically expanding the model capacity when distribution shifts are detected.

Key features of our implementation:
- Uses Vision Transformer (ViT) as the base model
- Implements modular adapters with representation descriptors
- Autoencoder-based representation descriptors to detect distribution shifts
- Expandable weighting router for mixture of adapter outputs
- Sub-linear expansion rate

## Reproduction Instructions

1. Ensure you have Docker installed
2. Run the reproduction script: