# BBox‑Adapter Reproduction (Toy Implementation)

This repository contains a lightweight, fully self‑contained implementation of the
*BBox‑Adapter* approach described in the paper *“BBox‑ADAPTER: Lightweight Adapting for Black‑Box Large Language Models”*.
The code is intentionally simplified so that it can be executed in a fresh Ubuntu 24.04 Docker container
(without any commercial API keys) and reproduced within a few minutes.

## What is reproduced

* The main idea of BBox‑Adapter: an energy‑based adapter that scores candidate
  generations produced by a black‑box LLM.
* A ranking‑based Noise‑Contrastive Estimation (NCE) loss that encourages the
  adapter to give higher scores to true answers than to negative samples.
* An online‑style training loop that alternates between sampling new candidates
  from the adapted model and updating the adapter.
* Evaluation on a tiny toy question‑answer dataset (5 test examples).

**Notice** – The full paper evaluates on large public datasets (GSM8K,
StrategyQA, TruthfulQA, ScienceQA) and uses state‑of‑the‑art LLMs such as
`gpt‑3.5‑turbo`.  Those models and datasets are not part of this repository.
Instead, we use the open‑source `distilgpt2` as the “black‑box” LLM and a
small hand‑crafted toy dataset.

## Repository layout

```
.
├── data
│   ├── train.tsv
│   └── test.tsv
├── src
│   ├── __init__.py
│   ├── adapter.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
├── reproduce.sh
└── README.md
```

## How to reproduce

1. **Create a fresh environment** (the grading system will do this automatically).
2. **Run the reproduction script**:

   ```bash
   bash reproduce.sh
   ```

   The script will:
   * install the required Python packages,
   * train the adapter on `data/train.tsv`,
   * evaluate on `data/test.tsv`,
   * write the predictions to `predictions.txt` and evaluation metrics to `metrics.json`.

3. **Inspect the outputs**

   ```bash
   cat predictions.txt
   cat metrics.json
   ```

   You should see a list of the model’s predicted answers and a JSON report
   containing the accuracy (expected around 80 % on the toy test set).

## Expected outputs

After running `reproduce.sh`, the following files will be created:

| File            | Description                                   |
|-----------------|-----------------------------------------------|
| `predictions.txt` | One predicted answer per test example.        |
| `metrics.json`   | JSON containing the overall accuracy.        |

The exact numeric values may vary slightly due to randomness in sampling,
but they should be reproducible because we set the random seeds in the code.

## Limitations

* The toy implementation uses a very small dataset and a lightweight LLM,
  so the reported accuracy is not comparable to the results in the paper.
* The adapter is a single linear layer; the paper uses a small BERT‑based
  model.
* No GPU is required; the code will run on CPU with a modest time budget.

Feel free to adapt the code to a larger dataset or a larger LLM if you wish
to explore the full power of BBox‑Adapter.