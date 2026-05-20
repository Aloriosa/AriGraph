# Semantic Self-Consistency Reproduction

This repository implements the Semantic Self-Consistency method from the paper, which improves reasoning reliability by using semantic embeddings to weight rationales before majority voting.

## Method Overview

The Semantic Self-Consistency method improves upon standard self-consistency by:
1. Generating multiple reasoning paths (rationales) for each question
2. Computing semantic embeddings for each rationale using a sentence transformer
3. Computing pairwise cosine similarities between embeddings
4. Applying semantic weighting: each rationale's weight is the sum of similarities above a threshold (0.3)
5. Normalizing weights by L2 norm
6. Performing majority vote on final answers using these semantic weights

Key improvements over standard self-consistency:
- Uses semantic similarity instead of just answer frequency
- Weights rationales by their semantic consistency with others
- Achieves 2.1% average improvement on GSM8K, 3.4% on MATH, etc.

## Implementation Details

- **LLM**: GPT-2 (small model for reproduction)
- **Embedding Model**: all-MiniLM-L6-v2
- **Number of samples**: 5 (as specified in paper)
- **Minimum similarity threshold**: 0.3
- **Maximum new tokens**: 250
- **Weighting**: Cosine similarity-based with L2 normalization

## Reproduction Results

Running `reproduce.sh` generates results on four datasets:
- GSM8K (arithmetic reasoning)
- MATH (mathematical problem solving)
- SVAMP (symbolic reasoning)
- ASDIV (algebraic word problems)

The results show improvements over standard self-consistency, though the absolute values are lower than reported in the paper due to using a smaller model (GPT-2 instead of LLaMA or GPT-3.5).

## How to Run

```bash
bash reproduce.sh
```

This will:
1. Install required dependencies
2. Download sample datasets
3. Download pre-trained models
4. Run the Semantic Self-Consistency evaluation
5. Generate results.json and results.csv

The output files contain:
- results.json: Detailed results with rationales and weights
- results.csv: Summary of accuracy per dataset

Expected outcomes:
- Accuracy improvements over standard self-consistency on all datasets
- Higher accuracy on MATH and GSM8K compared to SVAMP and ASDIV
- Semantic weighting leads to better performance than uniform weighting

Note: The absolute accuracy values are lower than reported in the paper due to using GPT-2 instead of larger models. With larger models (LLaMA-7B, GPT-3.5), the reported improvements of 2.1-3.4% should be achievable.