# Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective"

This repository contains the complete reproduction of the paper "Challenges in Training PINNs: A Loss Landscape Perspective" by Pratik Rathore et al.

## Overview

This reproduction implements the key experiments from the paper comparing different optimizers for training Physics-Informed Neural Networks (PINNs) on three challenging PDE problems: convection, reaction, and wave PDEs.

The reproduction implements the following optimizers as described in the paper:
1. Adam (first-order)
2. L-BFGS (second-order)
3. Adam+L-BFGS (combined)
4. NysNewton-CG (proposed method)

## Implementation Details

The implementation follows the experimental setup from the paper:

- **PDE Problems**: 
  - Convection PDE: ∂u/∂t + β∂u/∂x = 0, β = 40
  - Reaction ODE: ∂u/∂t - ρu(1-u) = 0, ρ = 5
  - Wave PDE: ∂²u/∂t² - 4∂²u/∂x² = 0

- **Network Architecture**: 
  - MLP with 3 hidden layers
  - Tanh activation functions
  - Xavier initialization
  - Widths: 50, 100, 200, 400

- **Training Parameters**:
  - 41,000 total iterations
  - 10,000 residual points
  - 257 initial condition points
  - 101 boundary condition points

- **Optimizers**:
  - Adam: learning rate tuned on {10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹}
  - L-BFGS: default parameters (learning rate 1.0, memory 100)
  - Adam+L-BFGS: Adam for 1000, 11000, or 31000 iterations, then L-BFGS for remaining iterations
  - NysNewton-CG: after Adam+L-BFGS

## Reproduction Script

The `reproduce.sh` script:
1. Sets up the environment (Ubuntu 24.04 LTS with Python 3.10.12)
2. Installs required packages (PyTorch, NumPy, Matplotlib)
3. Runs experiments on all three PDEs with all four optimizers
4. Generates summary statistics and plots
5. Saves results in the output directory

## Expected Results

Running this script should reproduce the key findings from the paper:

1. **Loss Landscape Ill-conditioning**: The Hessian spectral density shows large outlier eigenvalues (>10⁴ for convection, >10³ for reaction, >10⁵ for wave), confirming the loss landscape is ill-conditioned.

2. **Optimizer Comparison**: 
   - Adam converges slowly due to ill-conditioning
   - L-BFGS converges faster than Adam
   - Adam+L-BFGS consistently outperforms Adam or L-BFGS alone
   - NysNewton-CG further improves upon Adam+L-BFGS

3. **Quantitative Results**:
   - Adam+L-BFGS achieves 14.2× smaller L2RE than Adam on convection
   - Adam+L-BFGS achieves 6.07× smaller L2RE than L-BFGS on wave
   - NysNewton-CG further reduces loss by factor >10

4. **Theoretical Validation**: The results validate the paper's theoretical claims about the benefits of combining first- and second-order methods.

## Repository Structure