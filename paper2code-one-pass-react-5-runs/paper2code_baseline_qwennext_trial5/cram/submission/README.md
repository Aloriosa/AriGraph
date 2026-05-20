# Reproduction: "Cramming 1568 Tokens into a Single Vector and Back Again"

This repository contains the complete implementation to reproduce the results from the paper "Cramming 1568 Tokens into a Single Vector and Back Again" by Yuri Kuratov et al.

## Overview

The paper explores the theoretical and practical limits of encoding long sequences of text tokens into a single vector representation using pre-trained language models (LLMs). The authors propose a method that uses trainable "[mem]" vectors that are prepended to the input sequence, then optimized via gradient descent to enable the LLM to reconstruct the original sequence from only the "[mem]" vectors.

The key finding is that a single 4096-dimensional vector can encode up to 1568 tokens from the Llama-3.1-8B model, achieving a compression ratio of 1568x without any loss in reconstruction quality.

## Reproduction Instructions

1. Ensure you have a machine with an NVIDIA GPU and Docker installed.

2. Build the Docker image: