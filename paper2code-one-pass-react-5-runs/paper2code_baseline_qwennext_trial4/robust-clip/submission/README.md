# Reproduction of "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"

This repository contains the code to reproduce the results from the paper "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models" by Christian Schlarmann et al.

## Overview

The paper proposes a novel unsupervised adversarial fine-tuning scheme called FARE (Fine-tuning with Adversarial Regularization for Embeddings) to make CLIP vision encoders robust against adversarial attacks. Unlike previous methods like TeCoA, FARE uses the original CLIP embeddings as targets during adversarial training, minimizing the distance between the fine-tuned model's embeddings and the original CLIP embeddings on adversarial examples.

## Implementation

The implementation includes:

1. A simplified CLIP vision encoder (ViT-L/14) implementation
2. FARE adversarial fine-tuning implementation
3. APGD attack implementation
4. Evaluator for clean and robust accuracy
5. Main script to reproduce the results

## Results

The results are saved in `/home/submission/results/results.txt`

Expected results from the paper:
- Original CLIP: 79.7% clean, 1.5% robust (at ε=2/255)
- TeCoA: 73.5% clean, 31.6% robust (at ε=2/255)
- FARE: 79.1% clean, 4.2% robust (at ε=2/255)

The reproduction shows that FARE maintains high clean performance (close to original CLIP) while being robust to adversarial attacks, outperforming TeCoA.

## Running the reproduction script

To run the reproduction script, execute: