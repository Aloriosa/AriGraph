# Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective"

This repository contains the complete implementation to reproduce the results from the paper "Challenges in Training PINNs: A Loss Landscape Perspective" by Rathore et al.

## Overview

The paper investigates the challenges in training Physics-Informed Neural Networks (PINNs) and proposes improved optimization strategies. The key findings are:

1. The PINN loss landscape is ill-conditioned due to differential operators in the residual term
2. First-order methods like Adam converge slowly on this ill-conditioned landscape
3. Second-order methods like L-BFGS improve conditioning significantly
4. The combination Adam+L-BFGS outperforms either method alone
5. We introduce NysNewton-CG (NNCG) which further improves results

## Implementation

We have implemented the following components:

1. **PINN architecture**: Multi-layer perceptron with tanh activations for solving PDEs
2. **PDE solvers**: Implementation of convection, reaction, and wave equations from the paper
3. **Optimizers**: 
   - Adam (first-order)
   - L-BFGS (second-order)
   - Adam+L-BFGS (hybrid)
   - NysNewton-CG (novel second-order method)
4. **Training pipeline**: Complete training loop with loss computation and optimization
5. **Evaluation**: L2 relative error calculation against ground truth
6. **Visualization**: Plots of loss curves, Hessian spectra, and solution comparisons

## Reproduction Instructions

1. Ensure you have Docker installed with NVIDIA Container Toolkit
2. Run the reproduction script: