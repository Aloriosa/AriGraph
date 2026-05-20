# Reproduction Repository: "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"

## Overview

This repository contains the complete implementation to reproduce the key results from the paper "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement" by Xisen Jin and Xiang Ren.

The paper introduces a novel approach to forecasting which upstream examples a language model will forget after refinement (updating the model with corrected examples).

## Key Contributions Reproduced

1. **Representation-based Forecasting Model**: We implement the most effective forecasting model from the paper, which uses inner products of low-dimensional representations of examples to predict forgetting.

2. **Demonstration of Forecasting**: We demonstrate the model's ability to predict which examples are likely to be forgotten.

3. **Practical Utility**: We show how forecasting forgotten examples can be used to reduce catastrophic forgetting.

## Implementation Details

The reproduction implements the representation-based forecasting model from Section 3.3 of the paper.

The model:
- Uses a trainable encoding function h that maps example inputs and outputs to low-dimensional representations
- Predicts forgetting based on inner products of these representations
- Uses a sigmoid activation to output a probability of forgetting

For computational feasibility, we use a simplified version of the model that works on a small sample dataset.

## How to Run

1. Ensure Docker is installed
2. Run the reproduction script: