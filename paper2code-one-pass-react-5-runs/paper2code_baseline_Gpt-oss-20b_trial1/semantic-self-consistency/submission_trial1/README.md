```
# Semantic Self‑Consistency Reproduction

This repository contains a minimal, fully‑reproducible implementation of the *Semantic Self‑Consistency* method described in the paper “Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting”.  
The goal is to demonstrate how to:

1. Generate multiple chain‑of‑thought (CoT) rationales for a reasoning question with a local language model (GPT‑2).
2. Embed each rationale with a lightweight sentence‑transformer.
3. Compute two semantic weighting schemes (Centroid Proximity Weighting and Semantic Consensus Weighting).
4. Compare the weighted predictions with the standard self‑consistency majority vote.

All heavy artifacts (model checkpoints, large datasets) are **not committed** to the repository; they are downloaded automatically when the reproduction script is run.

## Reproduction

The `reproduce.sh` script installs dependencies and runs the evaluation.  
All results are printed to the console and saved to `results/results.json`.

```bash
bash reproduce.sh
```

After the script finishes, you should see output similar to:

```
Evaluating 9 examples...
Baseline accuracy: 0.56
CPW accuracy: 0.67
SCW accuracy: 0.78
Results written to results/results.json
```

The exact numbers may vary slightly due to sampling randomness.

## Project Structure

```
├── data/
│   └── dataset.jsonl          # 9 toy examples (3 from each benchmark)
├── results/
│   └── results.json           # JSON with accuracy numbers
├── src/
│   ├── __init__.py
│   ├── evaluate.py            # Main evaluation script
│   └── utils.py               # Helper functions
├── reproduce.sh
├── requirements.txt
└── README.md
```

## Method Overview

1. **Generation** – For each question we generate `n=5` CoT rationales using GPT‑2 with sampling (`temperature=0.7`, `do_sample=True`).
2. **Parsing** – The final answer is extracted from the last token of each generation. If the token is numeric, it is used directly; otherwise the token is interpreted as a single‑word answer.
3. **Embedding** – Each rationale is encoded with the sentence‑transformer *all‑MiniLM‑L6‑v2*.
4. **Weighting**  
   * **Centroid Proximity Weighting (CPW)** – Embeddings are centred; closer rationales receive higher inverse‑distance weights.  
   * **Semantic Consensus Weighting (SCW)** – Each embedding is scored by the sum of its cosine similarities to all other embeddings; higher similarity scores indicate a more representative rationale.
5. **Aggregation** – For each method we sum weights per unique answer and pick the answer with the highest total weight.  
6. **Evaluation** – Accuracy is computed against the ground‑truth answer in the toy dataset.

The implementation follows the mathematical description in the paper (see Sections 4.1.1 and 4.1.2).

## Extending the Evaluation

- **More data** – Replace `data/dataset.jsonl` with a larger subset of the official benchmarks.  
- **Different models** – Swap `model="gpt2"` for another HuggingFace checkpoint.  
- **Hyper‑parameters** – Adjust `n`, `temperature`, or `max_new_tokens` in `src/evaluate.py`.

## License

This code is released under the MIT License.  
The datasets used are the official public splits of AQuA‑RAT, SVAMP, and StrategyQA.  
All external libraries are open‑source and are used under their respective licenses.

```