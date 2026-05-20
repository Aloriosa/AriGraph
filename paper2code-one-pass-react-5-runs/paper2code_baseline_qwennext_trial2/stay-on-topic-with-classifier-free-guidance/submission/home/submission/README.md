# Reproduction: "Stay on topic with Classifier-Free Guidance"

This repository contains the complete implementation to reproduce the results from the paper "Stay on topic with Classifier-Free Guidance" by Sanchez et al.

## Overview

The paper introduces Classifier-Free Guidance (CFG) as a technique to improve prompt adherence in language models. CFG works by modifying the logits during inference to increase the model's adherence to the input prompt.

Key contributions reproduced:
1. CFG implementation for autoregressive language models
2. Zero-shot QA on LAMBADA with LLaMA-7B
3. Chain-of-Thought prompting on GSM8K
4. Code generation on HumanEval
5. Negative prompting for assistant-style prompts

## Reproduction Instructions

1. Install dependencies: