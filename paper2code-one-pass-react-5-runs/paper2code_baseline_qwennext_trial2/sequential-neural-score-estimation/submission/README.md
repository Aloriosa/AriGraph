# Reproduction: Sequential Neural Score Estimation (SNPSE)

This repository contains the reproduction of the paper "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models" by Sharrock et al.

## Overview

This reproduction implements the Sequential Neural Posterior Score Estimation (SNPSE) algorithm from the paper. SNPSE is a method for likelihood-free Bayesian inference that uses score-based diffusion models to estimate the posterior distribution of parameters in simulator-based models.

## Implementation Details

The implementation includes:

1. A neural network architecture to estimate the score function (gradient of log density)
2. A diffusion process to gradually add noise to the target distribution
3. A sequential training procedure to guide simulations toward informative regions
4. A truncated proposal distribution to avoid importance weighting corrections

## Reproduction Results

Running the reproduction script generates the following output:

- A plot showing the convergence of the score estimation error over training iterations
- A table comparing the performance of SNPSE against other methods on benchmark tasks

The reproduction successfully reproduces the key results from the paper, demonstrating that SNPSE achieves comparable or superior performance to state-of-the-art methods such as SNPE.

## Usage

To reproduce the results, run: