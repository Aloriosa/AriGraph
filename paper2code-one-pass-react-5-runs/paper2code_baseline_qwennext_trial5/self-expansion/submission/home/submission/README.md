# SEMA: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

This repository contains the complete reproduction of the paper "Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning" by Wang et al.

## Overview

The paper proposes SEMA (Self-Expansion of pre-trained models with Modularized Adaptation), a novel approach for continual learning that automatically expands pre-trained models with modularized adapters on demand. SEMA uses representation descriptors as distribution shift indicators to trigger expansion signals and an expandable weighting router for composing adapter outputs.

## Key Contributions

1. **Self-Expansion Mechanism**: Automatically decides when and where to expand the model based on distribution shifts detected by representation descriptors.

2. **Modular Adapters**: Each adapter consists of a functional adapter and a representation descriptor that captures the feature distribution.

3. **Expandable Weighting Router**: Learns to create a weighted mixture of adapter outputs.

4. **Sub-linear Expansion Rate**: Achieves sub-linear parameter growth rate compared to linear growth in existing methods.

## Reproduction Instructions

To reproduce the results from the paper, follow these steps:

1. **Install Dependencies**: