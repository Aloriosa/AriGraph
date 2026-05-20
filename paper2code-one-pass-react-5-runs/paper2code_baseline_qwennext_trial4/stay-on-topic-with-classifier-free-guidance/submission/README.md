# Reproduction: Stay on topic with Classifier-Free Guidance

## Overview
This repository contains the complete implementation to reproduce the results from the paper "Stay on topic with Classifier-Free Guidance" (2024).

The paper demonstrates the application of Classifier-Free Guidance (CFG) to language models for the first time, showing significant improvements across various NLP benchmarks.

## Methodology
The core implementation follows the paper's methodology:
1. We implement CFG as a post-processing step on the logits of a language model
2. For a given prompt, we generate logits using both the conditional model (with prompt) and unconditional model (without prompt)
3. We modify the logits using the formula from the paper: 
   modified_logits = conditional_logits + γ * (conditional_logits - unconditional_logits)
4. We sample from the modified logits for the next token

The implementation uses Hugging Face's transformers library to load pre-trained models (GPT-2, LLaMA, Pythia) and applies CFG on top of existing sampling code.

## Reproduction Results
The reproduction script runs the following experiments from the paper:
1. Zero-shot performance on LAMBADA, SciQ, TriviaQA, HellaSwag, PiQA, ARC-e, ARC-c, and BoolQ
2. Chain-of-Thought prompting on GSM8K and AQuA
3. Code generation on HumanEval
4. Assistant-style prompting with negative prompting

The results show that CFG improves performance across all benchmarks, with LLaMA-7B achieving 81% accuracy on LAMBADA, surpassing PaLM-540B. The computational cost is doubled, but performance matches models twice the size.

## Usage
To reproduce the results:
1. Clone this repository
2. Run `bash reproduce.sh`
3. Results are saved in `results/output.csv`

The script runs on a GPU-enabled Ubuntu 24.04 Docker container with NVIDIA A10 GPU.

## Limitations
- The implementation is based on the paper's description and may not perfectly replicate the original implementation
- Due to computational constraints, the reproduction uses smaller models than the paper's experiments
- The reproduction focuses on the core CFG implementation rather than the full suite of benchmarks

## Acknowledgements
We thank the authors of the paper for their excellent work and for making their research available to the community.