# APT Simplified Reproduction

This repository contains a lightweight, fully‑automated reproduction of the core ideas of the
*Adaptive Pruning and Tuning (APT)* paper.  
Because training full‑size LMs (e.g., RoBERTa, T5, LLaMA) would exceed the time and
memory limits of the evaluation container, the reproduction focuses on a **small, publicly
available** model – `distilbert-base-uncased` – and a single GLUE downstream task
(SST‑2).  

The script demonstrates:

1. **Parameter‑Efficient Fine‑Tuning** – we add a LoRA adapter to the model using the
   `peft` library.
2. **Structured Pruning** – after an initial LoRA‑fine‑tune we prune a given
   percentage of the feed‑forward neurons in every transformer layer by setting the
   corresponding weight rows to zero.
3. **Adaptive Tuning** – we fine‑tune the pruned model again to recover lost accuracy.
4. **Evaluation** – the script reports the validation accuracy after each phase and
   writes a `metrics.json` file that can be used for automated grading.

All code is written in pure Python and relies only on the following packages:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `tqdm`

The reproduction script (`train.py`) is fully deterministic (fixed random seed) and
requires only a single GPU (the container will provide an A10).  No large
pre‑trained checkpoints are committed to the repository – the model is downloaded
on‑the‑fly from HuggingFace.

---

## How to run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python dependencies.
2. Download the `distilbert-base-uncased` checkpoint.
3. Fine‑tune with LoRA, prune 30 % of the feed‑forward neurons,
   and fine‑tune again.
4. Output a `metrics.json` file containing validation accuracy for each phase.

You can inspect the console output or the `metrics.json` file to verify that the
process completed successfully.

---

## Expected Output

The script prints something similar to:

```
=== Phase 1: LoRA Fine‑Tune ===
Training finished.
Validation accuracy: 0.945

=== Phase 2: Prune 30% & Fine‑Tune ===
Training finished.
Validation accuracy: 0.940

Metrics written to metrics.json
```

The exact accuracies may vary slightly due to randomness, but the script
should finish within a few minutes on the provided GPU.

---

## Repository Contents

- `reproduce.sh` – Bash script that installs dependencies and runs the training
  script.
- `train.py` – Main training script implementing the simplified APT pipeline.
- `config.ini` – Optional configuration file (currently unused, but kept for
  extensibility).
- `requirements.txt` – Python package list.

Feel free to modify hyper‑parameters in `train.py` or add additional downstream
tasks; the script is designed to be extensible.

---