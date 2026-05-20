# Reproduction of "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"

## Overview

This repository contains the implementation of the Lexicographic Bilevel Coreset Selection (LBCS) algorithm proposed in the paper "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints".

The paper introduces a novel problem called "Refined Coreset Selection" (RCS), which aims to find the minimal coreset size under model performance constraints. The LBCS algorithm is designed to optimize two objectives in a lexicographic order: model performance (primary objective) and coreset size (secondary objective).

## Implementation Details

The implementation follows the algorithm described in Section 3 of the paper. Key components include:

1. **LeNet Architecture**: A simplified LeNet neural network for MNIST classification.
2. **LBCS Algorithm**: A randomized direct search algorithm that performs lexicographic optimization over the two objectives.
3. **Lexicographic Optimization**: The algorithm maintains an incumbent solution and iteratively improves it based on lexicographic comparisons between candidate solutions.

The implementation is designed to be computationally efficient and runs on CPU or GPU (if available).

## Reproduction Instructions

1. Clone this repository.
2. Navigate to the repository directory.
3. Run the reproduction script: