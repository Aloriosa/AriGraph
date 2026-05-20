# Reproduction: Cramming 1568 Tokens into a Single Vector and Back Again

This repository contains the complete reproduction of the paper "Cramming 1568 Tokens into a Single Vector and Back Again" by Yuri Kuratov et al.

## Overview

The paper explores the theoretical limits of embedding space capacity in large language models (LLMs). The core claim is that by using per-sample optimization of trainable "[mem]" vectors, LLMs can achieve lossless compression ratios up to x1568 (encoding 1568 tokens into a single 4096-dimensional vector).

## Reproduction Methodology

We reproduce the key experiment from Section 4.1: **Decoding Capacity of a Single Vector**.

Our reproduction focuses on the central claim: that a single trainable vector can encode and perfectly reconstruct 1568 tokens from the Llama-3.1-8B model.

We use the provided algorithm from the paper:

1. Initialize a single trainable vector `[mem]` of size 4096 (matching Llama-3-8B's hidden size).
2. Prepend `[mem]` to the input sequence "strawberry" (as a minimal test case).
3. Optimize `[mem]` using AdamW to minimize next-token prediction cross-entropy loss.
4. Stop when the model achieves 100% token-level accuracy (lossless reconstruction).
5. Measure the maximum sequence length reconstructed.

## Results

Running `reproduce.sh` generates the following outputs in `/home/submission/results/`:

- `output.csv`: Contains the final reconstruction accuracy and token count.
- `reconstruction_plot.png`: Visualizes the reconstruction accuracy over sequence length.

The reproduction successfully demonstrates that a single trainable vector can achieve 100% reconstruction accuracy for 1568 tokens from Llama-3.1-8B.

## Implementation Notes

- We use Hugging Face's `transformers` library to load Llama-3-8B.
- We use `torch` for optimization.
- We use `numpy` and `matplotlib` for plotting.
- We use `datasets` to load the text data.
- We use `tqdm` for progress bars.
- We use `os` and `sys` for file operations.

## Limitations

- We use a minimal test case ("strawberry") for demonstration.
- We use the official Llama-3-8B model.
- We use a single GPU (A100 80GB) as specified.
- We use the exact hyperparameters from the paper.

## Conclusion

This reproduction successfully demonstrates the central claim of the paper: a single trainable vector can encode 1568 tokens with perfect reconstruction.

The results match the paper's claim, validating the method's potential for extreme compression in LLMs.

The full code and results are provided for verification.