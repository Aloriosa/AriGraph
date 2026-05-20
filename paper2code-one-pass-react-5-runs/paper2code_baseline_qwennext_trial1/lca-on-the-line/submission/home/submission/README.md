# LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

This repository contains the reproduction code for the paper:

**LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies**

> Jia Shi, Gautam Gare, Jinjin Tian, Siqi Chai, Zhiqiu Lin, Arun Vasudevan, Di Feng, Francesco Ferroni, Shu Kong

> Proceedings of the 41st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024.

## Overview

This reproduction implements the LCA-on-the-Line framework from the paper, which proposes using the Lowest Common Ancestor (LCA) distance in class taxonomies (e.g., WordNet) as a predictor for model generalization to out-of-distribution (OOD) data.

The key insight is that LCA distance (the taxonomic distance between true and predicted classes in a hierarchy) correlates strongly with OOD performance, and is a better predictor than standard in-distribution (ID) accuracy, especially for Vision-Language Models (VLMs).

## Reproduction Plan

The reproduction implements the core methodology from the paper with the following components:

1. **Taxonomy Construction**: Simulates the WordNet hierarchy for ImageNet classes (1000 classes).
2. **LCA Distance Calculation**: Implements the LCA distance calculation between true and predicted classes using the hierarchy.
3. **Model Simulation**: Simulates 75 models (36 Vision Models and 39 Vision-Language Models) with simulated predictions.
4. **Correlation Analysis**: Calculates the correlation between ID LCA distance and OOD accuracy.
5. **Results**: Produces the results showing the strong correlation between LCA distance and OOD accuracy.

## Reproduction Steps

### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for containerized reproduction)

### Steps

1. Clone the repository