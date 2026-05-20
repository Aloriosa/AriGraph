# Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting

This repository contains the complete reproduction of the paper "Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting" by Tim Knappe et al.

## Overview

The paper introduces Semantic Self-Consistency, an enhancement to the self-consistency framework for improving reasoning in language models. The key contributions are:

1. **Centroid Proximity Weighting (CPW)**: A method that weights model responses based on their semantic similarity to the centroid of all generated rationales.

2. **Semantic Consensus Weighting (SCW)**: A method that uses cosine similarity to weight responses based on their consensus with other responses.

3. **Outlier Detection**: Methods to filter out degenerate or hallucinated outputs using k-nearest neighbor, isolation forest, and one-class SVM.

## Reproduction Instructions

To reproduce the results from the paper, follow these steps:

1. Clone this repository
2. Run the `reproduce.sh` script: