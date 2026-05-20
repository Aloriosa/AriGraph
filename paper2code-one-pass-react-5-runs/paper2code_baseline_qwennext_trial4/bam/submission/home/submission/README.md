# Reproduction of "Batch and match: black-box variational inference with a score-based divergence"

This repository contains a complete reproduction of the paper "Batch and match: black-box variational inference with a score-based divergence" by Cai et al.

## Overview

The paper introduces a novel algorithm called Batch and Match (BaM) for black-box variational inference. Unlike traditional methods that optimize the Evidence Lower Bound (ELBO), BaM uses a score-based divergence that can be optimized with a closed-form proximal update for Gaussian variational families.

The key contributions of the paper are:
1. Introduction of a new score-based divergence for variational inference
2. Development of the BaM algorithm that alternates between "batch" and "match" steps
3. Theoretical analysis showing exponential convergence for Gaussian targets
4. Empirical evaluation showing BaM converges faster than leading methods

## Reproduction Instructions

### Prerequisites
- Ubuntu 24.04 LTS
- NVIDIA A10 GPU
- Docker (with NVIDIA container toolkit)

### Setup
1. Clone this repository
2. Ensure Docker is running
3. Run the reproduction script: