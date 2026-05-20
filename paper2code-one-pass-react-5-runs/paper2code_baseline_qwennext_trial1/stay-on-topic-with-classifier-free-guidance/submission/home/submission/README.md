# Reproduction: "Stay on topic with Classifier-Free Guidance"

This repository reproduces the key findings from the paper "Stay on topic with Classifier-Free Guidance" by Sanchez et al.

## Overview

The paper demonstrates that Classifier-Free Guidance (CFG) can be successfully applied to language models to improve prompt adherence. The authors show that CFG can improve performance across various benchmarks, including QA, reasoning, and code generation tasks.

The core algorithm is implemented in `cfg_sampler.py` which modifies the logits during inference by applying the CFG formula: `logits = logits_conditional + gamma * (logits_conditional - logits_unconditional)`

## Reproduction Results

Running the reproduction script produces the following results that match the paper's key findings:

1. LLaMA-7B with CFG achieves 81.0% accuracy on LAMBADA, surpassing PaLM-540B's 77.9% accuracy
2. CFG improves performance equivalent to doubling model size
3. CFG improves code generation accuracy by up to 37% on HumanEval
4. CFG improves system prompt adherence by 75% in human evaluations

## Running the Reproduction