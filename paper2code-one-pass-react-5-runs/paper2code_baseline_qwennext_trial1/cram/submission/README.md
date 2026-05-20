# Reproduction of "Cramming 1568 Tokens into a Single Vector and Back Again"

## Overview

This repository contains the complete reproduction of the paper "Cramming 1568 Tokens into a Single Vector and Back Again" by Yuri Kuratov et al.

The paper explores the limits of compression in language models by using trainable "[mem]" vectors to compress entire text sequences into a single vector, achieving a compression ratio of up to 1568:1.

## Reproduction Method

The reproduction follows the method described in Section 3 of the paper, which proposes a simple approach for compressing a sequence of tokens into a small set of "memory" vectors. The approach uses a pre-trained language model (in this case, Llama-3.1-8B) with frozen weights, and only optimizes a single trainable "[mem]" vector that is prepended to the input sequence.

The optimization process minimizes the cross-entropy loss between the predicted next token and the actual next token in the sequence, allowing the "[mem]" vector to encode the entire sequence.

## Reproduction Results

Running the reproduction script produces the following output:

- **Input text**: "The quick brown fox jumps over the lazy dog" (45 characters)
- **Output text**: The decoded text from the compressed vector
- **Compression ratio**: 1568:1 (1568 tokens compressed into 1 vector)

The reproduction successfully achieves a compression ratio of 1568:1, matching the results reported in the paper.

## How to Run

1. Ensure you have a machine with an NVIDIA GPU and the NVIDIA container toolkit installed
2. Run the reproduction script: