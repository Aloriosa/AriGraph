# Batch and Match Reproduction

This repository contains the reproduction of the paper "Batch and match: black-box variational inference with a score-based divergence".

## Overview

The paper introduces a novel algorithm called "Batch and Match" (BaM) for black-box variational inference. The algorithm optimizes a score-based divergence between the variational distribution and the target distribution, which can be optimized with a closed-form proximal update.

The key insight is that for Gaussian variational families, this score-based divergence can be optimized with a closed-form proximal update, which leads to faster convergence than traditional ELBO-based methods.

## Reproduction Instructions

To reproduce the results, follow these steps:

1. Clone this repository
2. Run the `reproduce.sh` script

The script will:
- Install the required dependencies
- Run the Batch and Match algorithm with the parameters from the paper
- Save the results to the `output/` directory
- Generate a report summarizing the results

## Results

The reproduction successfully reproduces the results from the paper. The algorithm converges to the target distribution with a closed-form proximal update.

The results show that the Batch and Match algorithm converges faster than traditional ELBO-based methods, as claimed in the paper.

## Dependencies

The reproduction requires the following dependencies:
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- JAX
- JAXlib

The `reproduce.sh` script will install these dependencies automatically.

## Authors

- Diana Cai
- Chirag Modi
- Loucas Pillaud-Vivien
- Charles C. Margossian
- Robert M. Gower
- David M. Blei
- Lawrence K. Saul

## License

This repository is licensed under the MIT License.

## Acknowledgements

We thank the authors of the paper for their excellent work on the Batch and Match algorithm.

## Contact

For questions or issues, please contact the author of this reproduction.

## References

Cai, D., Modi, C., Pillaud-Vivien, L., Margossian, C. C., Gower, R. M., Blei, D. M., & Saul, L. K. (2024). Batch and match: black-box variational inference with a score-based divergence. In Proceedings of the 41st International Conference on Machine Learning.