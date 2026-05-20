# Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting

This repository contains the reproduction code for the paper:  
**"Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting"  
by Tim Knappe, Ryan Li, Ayush Chauhan, Kaylee Chhua, Kevin Zhu, Sean O'Brien (Algoverse AI Research)**

## Overview

This reproduction implements the **Semantic Self-Consistency** framework introduced in the paper. The paper proposes enhancing the standard Self-Consistency method for reasoning in LLMs by incorporating **semantic weighting** of reasoning paths.

Standard Self-Consistency samples multiple chain-of-thought responses, then selects the most frequent final answer.

Our implementation adds a **semantic weighting layer**:
1. Generate `n` chain-of-thought responses for a question.
2. Convert each full reasoning path into a **semantic embedding** using a BERT-based featurizer.
3. Apply **Centroid Proximity Weighting (CPW)** or **Semantic Consensus Weighting (SCW)** to assign weights to responses based on the similarity of their reasoning paths.
4. Aggregate weighted responses to select the final answer.

We reproduce the core algorithmic contribution: using semantic embeddings to weight reasoning paths before majority vote.

## Reproduction Instructions

### Prerequisites

- Python 3.9+
- NVIDIA GPU with at least 12GB VRAM
- Docker (optional, for guaranteed environment)

### Setup

1. Clone this repository: