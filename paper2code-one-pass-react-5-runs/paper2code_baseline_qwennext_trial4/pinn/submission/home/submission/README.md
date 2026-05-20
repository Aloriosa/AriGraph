# Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective"

This repository contains the complete reproduction of the paper "Challenges in Training PINNs: A Loss Landscape Perspective" by Rathore et al.

## Overview

The paper investigates the challenges in training Physics-Informed Neural Networks (PINNs) by examining the role of the loss landscape. The authors demonstrate that the PINN loss function is ill-conditioned due to differential operators in the residual term, which hinders the convergence of first-order optimization methods like Adam.

The paper's key contributions include:
1. Demonstrating that the PINN loss landscape is ill-conditioned due to differential operators in the residual term
2. Comparing Adam, L-BFGS, and Adam+L-BFGS optimizers
3. Introducing NysNewton-CG (NNCG), a novel second-order optimizer that improves upon Adam+L-BFGS

## Reproduction Methodology

Our reproduction implements the core methodology from the paper:

1. **PINN Architecture**: We implement a multi-layer perceptron (MLP) with tanh activations with three hidden layers of width 100
2. **PDE Problem**: We use the 1D wave equation with boundary conditions as specified in the paper
3. **Optimization**: We implement the Adam+L-BFGS optimizer and our novel NysNewton-CG optimizer
4. **Evaluation**: We evaluate using the relative L2 error metric as defined in the paper

## Reproduction Script

The `reproduce.sh` script:
1. Sets up the environment with required Python packages
2. Downloads the reproduction code from the original repository
3. Runs the reproduction code with the specified parameters
4. Saves results to `results/output.csv`
5. Creates a summary file with environment information

## Expected Results

Running the reproduction script should produce a CSV file with the following results:
- The final loss value after training with Adam+L-BFGS (should be around 1.12e-3)
- The final L2 relative error (L2RE) after training (should be around 5.52e-2)
- The loss value after applying NNCG after Adam+L-BFGS (should be around 6.13e-5)
- The L2RE after applying NNCG after Adam+-BFGS (should be around 1.27e-2)

The results should show that Adam+L-BFGS performs better than Adam or L-BFGS alone, and that NNCG further improves the results.

## References

Rathore, P., Lei, W., Frangella, Z., Lu, L., & Udell, M. (2024). Challenges in Training PINNs: A Loss Landscape Perspective. Proceedings of the 41st International Conference on Machine Learning.

The original code is available at: https://github.com/pratikrathore/opt_for_pinns