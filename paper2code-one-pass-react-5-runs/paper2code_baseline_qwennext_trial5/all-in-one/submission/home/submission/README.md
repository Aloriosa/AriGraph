# All-in-one Simulation-Based Inference: Simformer Reproduction

This repository contains a reproduction of the paper "All-in-one simulation-based inference" by Gloeckler et al. (2024).

## Overview

The paper introduces the Simformer, a novel method for amortized Bayesian inference that uses a probabilistic diffusion model with transformer architectures. The Simformer overcomes limitations of current methods by being able to handle function-valued parameters, missing data, and inference with unstructured data.

## Reproduction

This reproduction implements a simplified version of the Simformer using PyTorch. The implementation includes:

1. A transformer-based diffusion model for joint inference
2. A synthetic dataset based on the "Two Moons" example from the paper
3. Training and testing code
4. Visualization of results

## Running the Reproduction

To run the reproduction:

1. Ensure you have Python 3.8+ installed
2. Run the reproduction script: