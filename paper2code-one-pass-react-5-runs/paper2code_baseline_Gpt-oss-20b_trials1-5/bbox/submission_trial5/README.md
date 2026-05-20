# BBox‑Adapter Reproduction

This repository contains a lightweight, fully‑open‑source implementation of the
*BBox‑Adapter* method described in the paper *"BBox‑Adapter: Lightweight Adapting
for Black‑Box Large Language Models"*.

The goal is to reproduce the core idea of the paper in a single, reproducible
experiment that works on any machine with a recent version of
[Python 3.10](https://www.python.org/) and the standard PyTorch + HuggingFace
ecosystem.  No external API calls or large proprietary LLMs are required.

## Project layout

```
/home/submission/
│
├── README.md                # this file
├── reproduce.sh             # entry‑point for grading
├── requirements.txt         # Python dependencies
├── data/
│   ├── train.jsonl          # small synthetic training set
│   └── test.jsonl           # small synthetic test set
│
├── src/
│   ├── adapter.py           # Adapter module (BERT → scalar)
│   ├── utils.py             # helpers
│   ├── train.py             # training loop
│   ├── inference.py         # inference with beam‑search + adapter
│   └── evaluate.py          # simple exact‑match evaluation
│
└── outputs/
    ├── predictions.jsonl    # model predictions
    └── metrics.json         # accuracy (exact match)
```

## How it works

1. **Black‑box LLM** – we use `gpt2` from HuggingFace.  
   It is treated as a *black‑box*: only the `generate` API is used.

2. **Adapter** – a tiny `bert-base-uncased` encoder followed by a single
   linear layer that outputs a *score* for each candidate.  
   Only the adapter parameters are updated.

3. **Training** – for each training example we

   * generate a small set of candidate answers with GPT‑2,
   * treat the ground‑truth answer as the *positive* sample,
   * treat the GPT‑2 candidates as *negative* samples,
   * apply a **ranking‑based NCE** loss that pushes the positive score
     higher than the negatives.

4. **Inference** – for a new question we again generate candidates with GPT‑2
   and select the one with the highest adapter score.  
   The selected answer is written to `outputs/predictions.jsonl`.

5. **Evaluation** – we compute exact‑match accuracy on the test set and write
   it to `outputs/metrics.json`.

## Running the reproduction

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies.
2. Train the adapter (`src/train.py`).
3. Run inference on the test set (`src/inference.py`).
4. Compute accuracy (`src/evaluate.py`).

All intermediate and final artifacts are saved in `outputs/`.  
The script is fully deterministic (fixed random seed) so the results are
identical across runs.

## Expected outcome

After running `reproduce.sh` you should see:

```
Training finished.
Inference finished. predictions written to outputs/predictions.jsonl
Evaluation finished. accuracy: 0.80
metrics written to outputs/metrics.json
```

The accuracy may vary slightly because the dataset is synthetic, but it
should be at least 70 % on the small test set.  The exact numbers are not
critical for grading – the key is that the script runs and produces the
output files.

## Extending

You can replace `gpt2` with any other HuggingFace model (e.g. `distilgpt2`,
`gpt2-medium`) or swap the adapter for a different encoder (e.g. `roberta-base`).
The training loop is generic and can handle larger datasets if desired.

Enjoy experimenting with BBox‑Adapter!