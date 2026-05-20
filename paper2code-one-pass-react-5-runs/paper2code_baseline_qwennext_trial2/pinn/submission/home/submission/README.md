# Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective"

This repository contains the complete implementation to reproduce the key results from the paper "Challenges in Training PINNs: A Loss Landscape Perspective" by Rathore et al.

## Overview

The paper explores challenges in training Physics-Informed Neural Networks (PINNs) and proposes improved optimization strategies. The key findings are:

1. The PINN loss landscape is ill-conditioned due to differential operators
2. Adam+L-BFGS outperforms Adam or L-BFGS alone
3. NysNewton-CG (NNCG) improves upon Adam+L-BFGS

## Reproduction Implementation

This implementation reproduces the key experiments from the paper by:

1. Implementing a PINN for the 1D wave equation (as in Section A.3 of the paper)
2. Training with Adam, L-BFGS, Adam+L-BFGS, and Adam+L-BFGS+NNCG
3. Reporting final loss and L2RE (relative error) for each optimizer

## Running the Reproduction

To run the reproduction, execute: