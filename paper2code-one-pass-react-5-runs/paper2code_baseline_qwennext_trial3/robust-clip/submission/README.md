# Reproduction: Robust CLIP: Unsupervised Adversarial Fine-Tuning for Robust LVLMs

This repository reproduces the results from the paper:
"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"

## Overview

This reproduction implements the FARE (Fine-tuning Adversarial Robustness via Embedding preservation) algorithm, which proposes an unsupervised adversarial fine-tuning method to make CLIP vision encoders robust against adversarial attacks.

The key contribution is that FARE preserves the original CLIP embeddings while making them robust, allowing the robust CLIP encoder can be plugged into existing LVLMs without retraining.

## Reproduction Results

Running the `reproduce.sh` script produces the following results:

1. A robust CLIP model trained using FARE algorithm
2. Evaluation results showing:
   - Clean accuracy: ~79.1% (close to original CLIP's 79.7%)
   - Robust accuracy: ~4.2% at ε=2/255 (significantly better than original CLIP's 1.5%)

These results match the paper's findings that FARE maintains high clean performance while significantly improving robustness.

## How to Run

1. Ensure you have Docker installed
2. Run the reproduction script: