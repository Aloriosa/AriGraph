# Reproduction: A Mechanistic Understanding of Alignment Algorithms

This repository contains a reproduction of the paper "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity" by Andrew Lee et al.

## Overview

This reproduction implements the core concepts from the paper which studies how Direct Preference Optimization (DPO) affects toxicity in language models.

The paper's key findings are:
1. Toxicity is represented in language models through specific "toxic vectors" in MLP blocks
2. DPO doesn't remove toxicity capabilities but rather learns to bypass these toxic regions
3. This alignment can be undone by simply scaling key vectors to reactivate toxicity

## Reproduction Implementation

Our reproduction implements a simplified version of the paper's methodology:

1. **Toxic Vector Extraction**: We extract vectors that promote toxicity using a probe trained on the Jigsaw Toxic Comment Classification dataset.

2. **DPO Simulation**: We simulate the DPO algorithm's effect by learning an "offset" to bypass toxic regions.

3. **Un-alignment**: We demonstrate how to undo the alignment by scaling key vectors to reactivate toxicity.

## Running the Reproduction

To run the reproduction script: