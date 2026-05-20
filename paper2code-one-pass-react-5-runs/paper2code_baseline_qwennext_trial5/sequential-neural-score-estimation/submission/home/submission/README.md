# Reproduction of "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"

This repository contains the implementation to reproduce the results from the paper "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models" by Louis Sharrock et al.

## Overview

The paper introduces Sequential Neural Posterior Score Estimation (SNPSE), a score-based method for Bayesian inference in simulator-based models. The method leverages conditional score-based diffusion models to generate samples from the posterior distribution of interest.

## Implementation Details

The implementation includes:

1. `main.py`: Main script implementing the NPSE and TSNPSE algorithms
2. `model.py`: PyTorch implementation of the score network architecture
3. `diffusion.py`: Implementation of forward and reverse diffusion processes
4. `data_loader.py`: Data loading utilities for benchmark tasks
5. `evaluation.py`: Evaluation utilities for C2ST scores
6. `summary.py`: Summary statistics generation
7. `reproduce.sh`: Reproduction script

## Reproduction Instructions

To reproduce the results:

1. Ensure you have Python 3.7+ and pip installed
2. Run the reproduction script: `bash reproduce.sh`

The script will:
- Install required dependencies
- Download benchmark data
- Run the NPSE algorithm on the benchmark tasks
- Generate output files

## Results

The reproduction produces the following output files:
- `results/output.csv`: Results from the benchmark tasks
- `results/summary.txt`: Summary statistics of the results

The results should show C2ST scores for the benchmark tasks, with scores close to those reported in the paper (lower scores indicate better performance).

The implementation follows the methodology described in the paper, using a score network with MLP embeddings for θ and x, and a sinusoidal embedding for time t.

Note: Due to the complexity of the algorithm and potential differences in random number generation, exact reproduction of the paper's results may not be possible, but the results should be comparable.

## References

- Sharrock, L., Simons, J., Liu, S., & Beaumont, M. (2024). Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models. Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024.

## Contact

For questions or issues, please contact the author at: l.sharrock@lancaster.ac.uk