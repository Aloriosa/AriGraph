# Reproduction of "All-in-one simulation-based inference" (Simformer)

This repository contains the complete implementation to reproduce the results from the paper "All-in-one simulation-based inference" by Gloeckler et al. (2024).

## Overview

The paper introduces the **Simformer** - a novel amortized Bayesian inference method that overcomes limitations of current simulation-based inference methods by using a probabilistic diffusion model with transformer architectures.

The Simformer provides an "all-in-one" inference method that can:
- Handle models with function-valued parameters
- Handle inference scenarios with missing or unstructured data
- Sample arbitrary conditionals of the joint distribution of parameters and data, including both posterior and likelihood

## Reproduction Instructions

To reproduce the results from the paper, follow these steps:

1. Clone this repository
2. Run the reproduction script