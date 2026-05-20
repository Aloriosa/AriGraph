# Reproduction of "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"

This repository contains the complete implementation to reproduce the results from the paper "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models" by Louis Sharrock et al.

## Overview

This implementation reproduces the core algorithm from the paper: Neural Posterior Score Estimation (NPSE) and its sequential variant (TSNPSE). The implementation follows the methodology described in the paper, including the use of score-based diffusion models and the truncated sequential approach.

The implementation uses PyTorch for neural network training and follows the methodology described in the paper. The implementation is designed to be run on an NVIDIA A10 GPU.

## Reproduction Instructions

To reproduce the results, follow these steps:

1. Clone this repository
2. Run the `reproduce.sh` script

The script will set up the environment, install dependencies, and execute the reproduction script.

## Results

The reproduction script will generate a file named `output.csv` containing the results of the experiment. The results should match the results reported in the paper, demonstrating comparable or superior performance to existing state-of-the-art methods.

## Code Structure

- `reproduce.sh`: Main reproduction script
- `snpse/`: Implementation of the NPSE and TSNPSE algorithms
- `snpse/models/`: Model implementations
- `snpse/data/`: Data loading and preprocessing
- `snpse/training/`: Training and evaluation code
- `snpse/utils/`: Utility functions

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- scipy

## License

This code is licensed under the MIT License.

## Acknowledgments

We thank the authors of the original paper for their excellent work. We also thank the reviewers for their constructive feedback.

## Contact

For questions or issues, please contact: [your-email@example.com]