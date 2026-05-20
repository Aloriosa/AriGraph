# Reproduction of "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"

## Overview

This repository contains the complete implementation to reproduce the results from the paper:

> "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"  
> Xiaobo Xia, Jiale Liu, Shaokun Zhang, Qingyun Wu, Hongxin Wei, Tongliang Liu

The paper introduces a novel problem called **Refined Coreset Selection (RCS)**, which aims to find the **smallest possible coreset** while maintaining **comparable model performance**. The proposed method, **Lexicographic Bilevel Coreset Selection (LBCS)**, formulates RCS as a bilevel optimization problem with lexicographic preferences.

## Reproduction Instructions

### Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA A100 GPU (or compatible)
- Docker (already installed in the container)
- Internet access

### Running the Reproduction Script

1. Clone this repository (already done as you're reading this file)
2. Ensure the container has access to the GPU
3. Run the reproduction script: