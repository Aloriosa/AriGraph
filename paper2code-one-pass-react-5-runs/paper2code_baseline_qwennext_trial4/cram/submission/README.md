# Reproduction of "Cramming 1568 Tokens into a Single Vector and Back Again"

## Overview

This repository contains a complete reproduction of the research paper "Cramming 1568 Tokens into a Single Vector and Back Again" by Yuri Kuratov et al.

The paper explores the limits of compression in large language models by replacing traditional encoders with per-sample optimization procedures. The key finding is that vectors with compression ratios up to x1500 exist, highlighting a two-order-of-magnitude gap between existing and practically attainable solutions.

## Reproduction Methodology

Our reproduction implements the core algorithm described in Section 3 of the paper:

1. **Text Compression**: We use a pre-trained LLM (Llama-3-8B) as a frozen encoder.
2. **Memory Vectors**: We optimize a small set of trainable "mem" vectors (1 to 16) that are prepended to the input sequence.
3. **Training**: The "mem" vectors are optimized using gradient descent to minimize the next-token prediction loss.
4. **Decoding**: The compressed vectors are used as the initial context for text generation.

We replicate the experimental setup described in Section 4, evaluating the decoding capacity on texts from the PG-19 dataset.

## Code Structure

- `reproduce.sh`: Main script to run the reproduction.
- `compress_text.py`: Main implementation of the text compression algorithm.
- `output/`: Directory for output files.

## How to Run

1. Ensure you have a GPU with at least 24GB VRAM and CUDA support.
2. Clone this repository.
3. Run the reproduction script: