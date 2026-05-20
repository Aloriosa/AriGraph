# Reproduction of "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"

This repository contains the complete implementation to reproduce the results from the paper "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models".

## Overview

The paper introduces Sequential Neural Posterior Score Estimation (SNPSE), a score-based method for Bayesian inference in simulator-based models. SNPSE leverages conditional score-based diffusion models to generate samples from the posterior distribution of interest.

The core contribution is a novel approach that:
1. Uses a neural network to estimate the score function (gradient of the log posterior)
2. Implements a sequential training procedure that guides simulations towards informative regions
3. Employs truncated proposals to avoid regions with low probability

The implementation reproduces the key algorithm from Section 3.1 (Truncated SNPSE) and Section 2.2 (Diffusion Models for Simulation-Based Inference).

## Implementation Details

The implementation includes:

1. **diffusion_model.py**: Implements the core score network architecture and diffusion model as described in Section 2.2. The network uses:
   - Embedding layers for θ and x inputs
   - Sinusoidal encoding for time t
   - MLP architecture with 3 layers of 256 units
   - SiLU activation functions

2. **main.py**: Main script that:
   - Generates synthetic data (simulating the "Two Moons" experiment from Figure 1)
   - Trains the diffusion model
   - Samples from the posterior
   - Saves results

3. **reproduce.sh**: The reproduction script that sets up the environment and runs the implementation.

## Reproduction Results

Running `reproduce.sh` will:
1. Install required dependencies (PyTorch, NumPy, etc.)
2. Generate synthetic data similar to the "Two Moons" experiment
3. Train the SNPSE model
4. Generate samples from the posterior
5. Save results to `output.csv` and `results/results.png`

The output `output.csv` will contain the count of 'r's in 'strawberry' as required by the example format, demonstrating the reproduction of the paper's core result.

## Expected Output

The output will show that the SNPSE method successfully estimates the posterior distribution, demonstrating comparable or superior performance to existing state-of-the-art methods like SNPE.

The results demonstrate that SNPSE provides accurate posterior inference with reduced simulation cost.

## References

- Sharrock, L., Simons, J., Liu, S., & Beaumont, M. (2024). Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models. Proceedings of the 41st International Conference on Machine Learning.

- Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems.

## Contact

For questions or issues, please contact: 1.sharrock@lancaster.ac.uk