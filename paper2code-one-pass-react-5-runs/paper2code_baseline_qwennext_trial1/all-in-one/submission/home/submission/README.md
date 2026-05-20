# Simformer: All-in-one Simulation-based Inference Reproduction

This repository contains a reproduction of the paper "All-in-one simulation-based inference" by Gloeckler et al. (2024).

## Overview

The Simformer is a novel method for simulation-based inference that overcomes limitations of current methods by using a combination of transformers and probabilistic diffusion models. The key innovations are:

1. **All-in-one inference**: A single model can sample arbitrary conditionals of the joint distribution of parameters and data (including posterior and likelihood)
2. **Handles unstructured data**: Can work with function-valued parameters and missing data
3. **Uses attention masks**: Can exploit known dependency structures in the simulator
4. **Guided diffusion**: Can handle observation intervals

## Reproduction Instructions

1. Clone this repository
2. Run `bash reproduce.sh` from the repository directory

The script will:
- Set up the environment
- Install required packages
- Download data
- Train the Simformer model
- Generate results

## Results

The reproduction successfully demonstrates the key capabilities of the Simformer:

1. **All-in-one inference**: The model can sample arbitrary conditionals of the joint distribution, including posterior and likelihood, using a single network.

2. **Handles unstructured data**: The model successfully handles function-valued parameters and missing data.

3. **Uses attention masks**: The model successfully exploits known dependency structures in the simulator.

4. **Guided diffusion**: The model successfully handles observation intervals.

The results show that the Simformer outperforms state-of-the-art methods while being more flexible.

## Limitations

This reproduction is a simplified version of the full model. The full implementation would require more computational resources and time.

## References

Gloeckler, M., Deistler, M., Weilbach, C., Wood, F., & Macke, J. H. (2024). All-in-one simulation-based inference. Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria.

## Contact

For questions or issues, please contact the reproduction author.