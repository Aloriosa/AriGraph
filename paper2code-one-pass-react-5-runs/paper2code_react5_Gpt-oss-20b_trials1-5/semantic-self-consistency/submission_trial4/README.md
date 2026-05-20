# Semantic Self‑Consistency Reproduction

This repository reproduces the core methods described in  
**“Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting”**.

## Contents

- `run_experiment.py` – Main pipeline that:
  1. Loads the three benchmark datasets (AQuA‑RAT, SVAMP, StrategyQA).
  2. Generates multiple rationales per question using a small GPT‑2 generator.
  3. Parses the final answer from each rationale.
  4. Applies three decision strategies:
     * **Baseline Self‑Consistency** – majority vote of final answers.
     * **Centroid Proximity Weighting (CPW)** – weights rationales by distance to the centroid of the embedding space.
     * **Semantic Consensus Weighting (SCW)** – weights rationales by aggregate cosine similarity.
  5. Computes accuracy and writes per‑example results to CSV files.
- `reproduce.sh` – Install dependencies and run the experiment.
- `requirements.txt` – Python packages required.

## Reproduction Procedure

1. **Clone the repository** (or copy the files to `/home/submission/` in the evaluation container).

2. **Run the reproduction script**:

   ```bash
   bash reproduce.sh
   ```

   The script will:
   - Install system packages and Python dependencies.
   - Download the required models (GPT‑2, SciBERT, RoBERTa) from HuggingFace.
   - Execute `run_experiment.py`, which processes all datasets and prints a summary.

3. **Inspect the results**:

   After completion you will find three CSV files:
   - `results_aquarat.csv`
   - `results_svamp.csv`
   - `results_strategyqa.csv`

   Each file contains the gold answer and the predictions from the three strategies.

4. **Summary output**:

   The script prints a tabular summary of accuracies for each dataset and method.

## What is reproduced

- **Baseline self‑consistency**: majority vote over multiple chain‑of‑thought samples.
- **Centroid Proximity Weighting (CPW)**: weighting by inverse distance to the centroid of the embedded reasoning paths.
- **Semantic Consensus Weighting (SCW)**: weighting by aggregated cosine similarity among reasoning paths.

The implementation intentionally uses *GPT‑2* and publicly available BERT variants to keep the repository lightweight and fully reproducible without additional GPU memory or proprietary API keys.

## Limitations

- The generator is a small GPT‑2 model, not the large LLMs used in the paper. Accuracy will be lower, but the pipeline demonstrates the algorithmic workflow.
- The embedding models are generic SciBERT and RoBERTa, not fine‑tuned as in the original paper. This affects clustering quality but preserves the conceptual structure.
- The script does not implement outlier removal or temperature‑weighted sampling variants; those can be added following the same pattern.

## License

This repository is released under the MIT license. Model weights and datasets are used under the licenses specified by HuggingFace.