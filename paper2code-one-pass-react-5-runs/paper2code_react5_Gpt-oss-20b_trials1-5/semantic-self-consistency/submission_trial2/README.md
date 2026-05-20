# Semantic Self‑Consistency Reproduction

This repository implements the core pipeline from the paper *Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting*.  
The reproduction follows the steps:

1. **Generation** – Sample `n` chain‑of‑thought responses from a large language model (default LLaMA‑2‑7B).  
2. **Embedding** – Encode each full rationale with a fine‑tuned SciBERT (or RoBERTa for StrategyQA) to obtain a semantic vector.  
3. **Weighting** – Compute two weighting schemes:
   * **Centroid Proximity Weighting (CPW)** – weight inversely proportional to distance from the mean embedding.  
   * **Semantic Consensus Weighting (SCW)** – weight proportional to summed cosine similarity with all other embeddings.  
4. **Prediction** – Perform weighted majority vote over the final answers.  
5. **Evaluation** – Compare against the baseline self‑consistency (unweighted majority vote) and report accuracy.

The script `reproduce.sh` downloads the required models, runs the full pipeline on a small demo dataset, and outputs a CSV with per‑example and aggregate results.

> **Note**: The demo dataset (`sample_dataset.jsonl`) contains only five toy problems.  
> For large‑scale experiments (full AQuA‑RAT, SVAMP, StrategyQA) replace the dataset path and increase `num_samples` in `config.json`.  
> The code can be extended to include outlier filtering (K‑NN, Isolation Forest, One‑class SVM) by adding the corresponding logic in `evaluate.py`.

## Reproducing the Results

```bash
bash reproduce.sh
```

After a few minutes (model download + first inference) you will find `results/run_results.csv`:

| question_id | true_answer | baseline_pred | baseline_acc | cpw_pred | cpw_acc | scw_pred | scw_acc |
|-------------|-------------|---------------|--------------|----------|---------|----------|---------|
| 0 | 12 | 12 | 1 | 12 | 1 | 12 | 1 |
| … | … | … | … | … | … | … | … |

The accuracy columns are 1.0 if the prediction matches the ground truth, 0 otherwise.

## Extending the Pipeline

- **Different Models** – Change the `--model` flag in `main.py` or the `model` field in `config.json`.  
- **More Samples** – Increase `num_samples` in `config.json`.  
- **Temperature** – Adjust `temperature`, `top_p`, `top_k`.  
- **Outlier Removal** – Implement filters in `evaluate.py` and toggle via `outlier_method` in `config.json`.  

## License

The code is released under the MIT license. Models are downloaded from HuggingFace and are subject to their respective licenses (e.g., LLaMA‑2 is under the LLaMA‑2 license, SciBERT under the Apache‑2.0 license).

## Contact

For questions or clarifications, please open an issue on the GitHub repository.