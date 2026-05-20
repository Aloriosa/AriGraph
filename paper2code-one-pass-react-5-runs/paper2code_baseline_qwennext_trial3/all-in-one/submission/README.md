# Reproduction of "All-in-one simulation-based inference" (Simformer)

This repository contains a reproduction implementation of the Simformer model from the paper "All-in-one simulation-based inference" by Gloeckler et al. (2024).

## Overview

The Simformer is a novel method for simulation-based amortized inference that uses a combination of transformers and probabilistic diffusion models to perform Bayesian inference on simulation models. The key innovation is that it can handle function-valued parameters, unstructured data, and can sample arbitrary conditionals of the joint distribution (posterior, likelihood, etc.) in a single unified framework.

## Implementation Details

This implementation provides a simplified but functional reproduction of the core components of the Simformer model:

1. **Simformer Architecture**: A transformer-based model that processes tokens representing both parameters and data variables
2. **Diffusion Model**: A score-based diffusion model that estimates the score function for the joint distribution
3. **Attention Masks**: Implementation of attention masks to encode dependency structures from the simulator
4. **Guided Diffusion**: Implementation of guidance to handle interval constraints

The implementation focuses on the core algorithmic components rather than the full computational scale of the original paper. The reproduction demonstrates the key capabilities: handling function-valued parameters, unstructured data, and sampling arbitrary conditionals.

## Reproduction Instructions

To reproduce the results from the paper, follow these steps:

1. Clone this repository
2. Run the reproduction script: