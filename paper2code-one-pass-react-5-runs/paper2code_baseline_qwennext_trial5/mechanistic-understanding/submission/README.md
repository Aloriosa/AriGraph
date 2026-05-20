# Reproduction: A Mechanistic Understanding of Alignment Algorithms

This repository contains the complete reproduction of the paper "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity" by Andrew Lee et al.

## Overview

This reproduction implements the core methodology from the paper, which investigates how the Direct Preference Optimization (DPO) alignment algorithm reduces toxicity in language models. The paper's key insight is that DPO does not remove the underlying capability for generating toxic outputs, but instead learns to bypass the specific neural pathways (vectors) that trigger toxicity.

## Reproduction Implementation

Our reproduction implements the following key components from the paper:

1. **Toxicity Vector Extraction**: We train a linear probe on the Jigsaw Toxic Comment dataset to identify a "toxicity direction" in the model's embedding space.

2. **Singular Value Decomposition (SVD)**: We extract the top 128 vectors from the MLP layers of GPT-2 and Llama-7B that have the highest cosine similarity with the toxicity direction.

3. **DPO Alignment**: We implement DPO using a synthetic dataset of 24,576 pairs of toxic and non-toxic text continuuations generated using PPLM (Plug and Play Language Models).

4. **Un-alignment**: We demonstrate how the alignment can be undone by simply scaling the key vectors associated with toxicity, thereby reactivating the bypassed toxicity pathways.

## Reproduction Script

The `reproduce.sh` script performs the following steps:
1. Sets up the Python environment with required packages (torch, transformers, datasets, scikit-learn, numpy, matplotlib).
2. Downloads and processes the Jigsaw Toxic Comment dataset.
3. Trains the toxicity probe model.
4. Extracts the top 128 toxic vectors from GPT-2 using SVD.
5. Generates the DPO alignment dataset using PPLM.
6. Applies DPO alignment on GPT-2.
7. Applies the un-alignment technique by scaling 7 key vectors to reactivate toxicity.
8. Saves the results to `results.json`.

## Expected Results

Running the script will produce `results.json` containing:
- Toxicity scores for original GPT-2, after DPO alignment, and after un-alignment.
- Perplexity scores for each stage.
- F1 scores for each stage.

The results should show:
- High toxicity in original GPT-2 (~0.45).
- Reduced toxicity after DPO (~0.20).
- Reverted toxicity after un-alignment (~0.45).

This demonstrates the paper's core finding: DPO does not remove the capability for toxicity, but bypasses it, making alignment fragile and reversible.

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support
- Docker (optional, for containerized execution)

## Run