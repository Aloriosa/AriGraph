# Semantic Self‑Consistency Reproduction

This repository contains a minimal, fully reproducible implementation of the *Semantic Self‑Consistency* paper (Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting).  
The goal is to demonstrate the core ideas:

1. **Self‑consistency** – sample `n` chain‑of‑thought rationales for each question and pick the majority final answer.
2. **Centroid Proximity Weighting (CPW)** – weight each rationale by its distance to the centroid of the embedding space.
3. **Semantic Consensus Weighting (SCW)** – weight each rationale by the sum of its cosine similarities to all other rationales.

The implementation uses open‑source models only:

* **LLM** – `meta-llama/Llama-2-7b-hf` (requires an HF token with Llama‑2 access).  
* **Featurizer** – `sentence-transformers/all-mpnet-base-v2` (fast, works for both math and QA).

The code runs on a single NVIDIA GPU (e.g., A10) and can be executed with a single shell command.

## How to run

```bash
# Install the dependencies
bash reproduce.sh
```

`reproduce.sh` will:

1. Install the required packages.
2. Run the full experiment for the three datasets in the paper:  
   * AQuA‑RAT  
   * SVAMP  
   * StrategyQA
3. Save the final results in `results.csv`.

The script prints a summary of the accuracies for each dataset and for each method.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── reproduce.sh
└── src
    ├── main.py
    └── utils.py
```

## Notes

* The paper also discusses outlier filtering (Isolation Forest, K‑NN, One‑class SVM).  
  For brevity, this implementation focuses on the core weighting methods.  
  Adding filtering is straightforward – see *utils.py* for a placeholder.
* The implementation uses the 8‑shot prompt provided in the paper’s Appendix K.  
  For datasets with a different prompt style, the code can be easily adapted.
* The code is intentionally simple and heavily commented to aid understanding.