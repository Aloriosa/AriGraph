# Reproduction: Refined Coreset Selection (RCS)

This repository contains the reproduction of the paper:

> **Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints**

## Overview

This reproduction implements the Lexicographic Bilevel Coreset Selection (LBCS) algorithm proposed in the paper. The algorithm addresses the problem of Refined Coreset Selection (RCS), which aims to find the minimal coreset size while maintaining comparable model performance.

## Implementation Details

The implementation follows the methodology described in Section 3 of the paper, specifically implementing Algorithm 1 (Lexicographic bilevel coreset selection for RCS).

Key components:
- **LeNet Architecture**: As used in the paper for MNIST experiments
- **Lexicographic Bilevel Optimization**: Implements the priority order where model performance (O1) is primary and coreset size (O2) is secondary
- **Randomized Search**: Uses a randomized direct search algorithm for the outer loop optimization

## Reproduction Results

Running the reproduction script executes the LBCS algorithm on the MNIST dataset with the following parameters:
- Coreset size: 100 (as used in the paper)
- Performance compromise (ε): 0.2
- Maximum search iterations: 50

The reproduction produces the following results:
- `results.pth`: PyTorch file containing the best masks, performance history, and coreset size history
- `results.png`: Visualization of the optimization progress showing:
  - Performance (loss) over iterations
  - Coreset size over iterations

The results demonstrate the algorithm's ability to find a coreset with minimal size while maintaining comparable performance to the full dataset.

## Dependencies

The reproduction requires the following Python packages:
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`

The `reproduce.sh` script automatically installs these dependencies.

## Usage

To reproduce the results, simply run: