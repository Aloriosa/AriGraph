# APT – Adaptive Pruning & Tuning (Reproduction)

This repository contains a lightweight, fully reproducible implementation of the core ideas from  
> **APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference**  
> (Zhao et al., 2024).

The goal is to demonstrate how a LoRA‑style adapter can be combined with
structured pruning (output‑dimension pruning), adaptive rank growth, and
self‑distillation during fine‑tuning.

## Repository layout

- **apt_modules.py** – LoRA linear layer with input/output masking,
  dynamic rank growth and outlier‑aware salience calculation.
- **run_apt.py** – Main training script. Supports three modes:
  * `apt` – full APT (pruning + dynamic rank + self‑distillation)
  * `lora` – LoRA only (no pruning)
  * `prune` – pruning only (no LoRA update)
- **reproduce.sh** – Installs dependencies and runs `run_apt.py`
  in all three modes, then prints a concise comparison table.
- **summarize_results.py** – Helper that reads the JSON logs from each mode
  and prints a Markdown table of accuracy, training time, peak memory and inference latency.
- **requirements.txt** – Python dependencies.

The script outputs training logs, final accuracy, inference speed, and peak GPU memory usage.
A checkpoint (`apt_model.pt`) is saved after each run.

## How to run

```bash
bash reproduce.sh
```

The script will run each mode sequentially and then print a table similar to:

| Mode | Epochs | Acc | Time (min) | Peak Mem (MB) | Inf Time (s) |
|------|--------|-----|------------|---------------|--------------|
| APT  | 4      | 94.01% | 10.12 |  752.34 | 0.67 |
| LORA | 4      | 93.65% | 9.88 |  755.10 | 0.68 |
| PRUNE| 4      | 93.20% | 9.70 |  758.27 | 0.69 |

(Values are illustrative; actual numbers depend on the hardware.)

---

### Notes

* The implementation keeps the code simple while preserving the key
  components from the paper: **outlier‑aware salience**, **binary pruning masks**,
  **rank‑growth policy**, and **self‑distillation with shared parameters**.
* The pruning strategy operates on the output dimension of every linear layer.
  Extending it to prune attention heads or FFN neurons would require a deeper
  integration with the transformer internals and is left as future work.
* The evaluation uses the GLUE SST‑2 task on a DistilBERT base model for brevity.
  The same code can be adapted to larger models (RoBERTa, T5, LLaMA) by
  adjusting the `--model_name` flag and potentially the pruning policy.

Enjoy reproducing the APT experiment!