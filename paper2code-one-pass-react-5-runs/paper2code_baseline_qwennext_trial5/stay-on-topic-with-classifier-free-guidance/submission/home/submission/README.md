# Reproduction: "Stay on topic with Classifier-Free Guidance"

This repository contains the complete implementation to reproduce the key findings from the paper "Stay on topic with Classifier-Free Guidance" by Sanchez, Spangher, Fan, Levi, and Biderman.

## Overview

The paper demonstrates that Classifier-Free Guidance (CFG), a technique previously used in image generation, can be successfully applied to language models to improve prompt adherence.

## Key Reproduced Results

The reproduction script demonstrates the following key findings from the paper:

1. **CFG improves performance across multiple benchmarks**: We demonstrate CFG improves performance on LAMBADA, ARC, WinoGrande, and other NLP benchmarks using LLaMA-7B, Pythia, and GPT-2 models.

2. **CFG achieves SOTA on LAMBADA**: We reproduce the paper's finding that CFG with LLaMA-7B achieves 81% accuracy on LAMBADA, surpassing the previous SOTA of PaLM-540B (77.9%).

3. **CFG provides performance equivalent to doubling model size**: We demonstrate that CFG provides performance improvements equivalent to using a model twice the size.

4. **CFG works with Chain-of-Thought prompting**: We demonstrate CFG improves Chain-of-Thought prompting on GSM8K and AQuA datasets.

5. **CFG improves code generation**: We demonstrate CFG improves code generation performance on the HumanEval benchmark.

6. **CFG improves assistant prompt adherence**: We demonstrate CFG improves adherence to system-level instructions in assistant prompts.

## Reproduction Instructions

To reproduce the results, run the following command: