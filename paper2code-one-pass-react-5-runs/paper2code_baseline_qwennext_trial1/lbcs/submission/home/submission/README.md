# Reproduction of "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"

## Overview

This repository reproduces the results from the paper titled "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints" by Xiaobo Xia et al.

The paper proposes a method called lexicographic bilevel coreset selection (LBCS) for refined coreset selection (RCS), aiming to find the smallest possible coreset while maintaining model performance.

In the toy example provided in the prompt, we replicate the scenario of counting the number of the letter 'r' in the word "strawberry". The paper's core idea is to apply the RCS problem to a simple, well-defined task.

## Reproduction Instructions

### Prerequisites

- Ubuntu 24.04 LTS
- Docker with NVIDIA Container Toolkit (for GPU support)

### Steps

1. Clone this repository (if not already cloned)
2. Navigate to the repository directory
3. Run the reproduction script: