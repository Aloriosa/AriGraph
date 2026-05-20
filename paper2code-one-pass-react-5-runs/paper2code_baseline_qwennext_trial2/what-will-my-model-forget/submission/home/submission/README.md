# Reproduction Repository: "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"

## Overview

This repository contains the complete implementation to reproduce the key findings from the paper "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement" by Jin and Ren (2024).

The paper introduces a novel approach to forecast which upstream examples will be forgotten during language model refinement. The core contributions include:
1. A novel problem setup of forecasting forgotten examples in model refinement
2. Two forecasting models: a partially interpretable logit-change based model and a black-box representation-based model
3. A model refinement algorithm with reduced forgetting by replaying examples predicted to be forgotten

## Implementation Details

This implementation reproduces the core forecasting models from Section 3.2 (Logit-Change based Forecasting) and Section 3.3 (Representation-Based Forecasting).

The implementation uses:
- PyTorch for deep learning framework
- scikit-learn for evaluation metrics
- NumPy for numerical computations
- A toy dataset simulating the P3 dataset described in the paper

## Reproduction Instructions

To reproduce the results, run the following commands: