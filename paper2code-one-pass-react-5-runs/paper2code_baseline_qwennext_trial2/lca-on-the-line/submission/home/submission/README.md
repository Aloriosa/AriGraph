# LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies

This repository contains the complete implementation to reproduce the results from the paper "LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies" (Shi et al., ICML 2024).

## Overview

The paper introduces the **LCA-on-the-Line** framework, a novel method for predicting models' Out-of-Distribution (OOD) performance using only in-distribution (ID) measurements. This framework uses the **Lowest Common Ancestor (LCA) distance** within a class taxonomy hierarchy (e.g., WordNet) as a robust predictor of OOD generalization performance.

The key findings of the paper are:

1. **The LCA distance is a superior predictor of OOD performance compared to ID accuracy.
2. The LCA distance reveals a strong linear correlation between ID LCA and OOD accuracy across 75 models (36 Vision Models and 39 Vision-Language Models).
3. The LCA distance unifies the evaluation of both Vision Models and Vision-Language Models under severe distribution shifts.

## Reproduction Instructions

To reproduce the results from the paper, follow these steps:

### Prerequisites

- Docker (optional, for containerized execution)
- Python 3.7+
- Git

### Step 1: Clone the Repository