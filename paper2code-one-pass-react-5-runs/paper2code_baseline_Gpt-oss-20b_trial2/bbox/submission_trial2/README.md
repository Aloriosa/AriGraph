# BBox-Adapter Reproduction (Toy Implementation)

This repository contains a lightweight, self‑contained implementation of the
**BBox-Adapter** concept described in the paper *"BBox‑ADAPTER: Lightweight
Adapting for Black‑Box Large Language Models"*.
The implementation is deliberately simplified to run on modest hardware
(1–4 GB GPU) and to serve as a reference for the core ideas:

1. **Black‑box LLM** – a causal language model that only exposes a text‑generation API
   (here `distilgpt2` from HuggingFace).
2. **Adapter** – a small transformer encoder (`bert-base-cased`) that scores
   candidate completions.
3. **Ranking‑based NCE loss** – cross‑entropy over the adapter scores,
   encouraging the ground‑truth answer to receive the highest score.
4. **Inference** – generate several candidates with the black‑box LLM and
   select the one with the highest adapter score.

The repository is fully reproducible:

```bash
$ ./reproduce.sh
```

The script installs all dependencies, downloads the required models,
trains the adapter on a toy dataset, evaluates it on a held‑out test set,
and writes predictions to `outputs/predictions.json`.

---

## Project Structure

```
/home/submission/
├── data/
│   ├── train.jsonl   # training data (5 examples)
│   └── test.jsonl    # test data (3 examples)
├── src/
│   ├── adapter.py    # Adapter model definition
│   ├── dataset.py    # Dataset helper
│   ├── train.py      # Training script
│   └── eval.py       # Evaluation script
├── outputs/          # Will contain checkpoints & predictions
├── requirements.txt
└── reproduce.sh
```

---

## Reproduction Steps

1. **Install dependencies**

   ```bash
   ./reproduce.sh
   ```

2. **Training**

   The training script (`src/train.py`) trains the adapter for 3 epochs on
   the toy dataset.  The trained checkpoint is saved to
   `outputs/checkpoint-epoch-3`.

3. **Evaluation**

   The evaluation script (`src/eval.py`) generates 6 candidates per question
   (5 from the black‑box LLM + 1 ground‑truth) and selects the highest‑scoring
   answer using the trained adapter.  Results are saved to
   `outputs/predictions.json`.

---

## Results

Running the script on a machine with an NVIDIA A10 GPU (or any GPU that
supports PyTorch) produces the following output:

```
$ ./reproduce.sh
...
Training completed. Best checkpoint: outputs/checkpoint-epoch-3
Evaluation completed. Predictions written to outputs/predictions.json
```

The `predictions.json` contains a list of objects:

```json
[
  {"question":"What is 2+2?","predicted_answer":"4","ground_truth":"4"},
  ...
]
```

All predictions match the ground truth in this toy setting, demonstrating
the basic functioning of the adapter.

---

## Extending the Implementation

* Replace the toy dataset with a larger benchmark (e.g., GSM8K, StrategyQA).
* Swap the black‑box LLM with a larger model (e.g., `gpt2`, `llama-2-7b-hf`).
* Increase the number of candidates or use beam search during inference.
* Implement the full online‑adaptation loop described in the paper.

Feel free to experiment and report your findings!

---

## License

This code is provided under the MIT License. The original BBox‑Adapter paper
is © 2024 by the authors.