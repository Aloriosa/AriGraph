```markdown
# APT Reproduction Repository

This repository contains a lightweight, fully‑reproducible implementation of the
*Adaptive Pruning and Tuning (APT)* idea presented in the paper
“APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient
Training and Inference”.  
The goal is to demonstrate a working pipeline that

1. **Adds a dynamic LoRA‑style adapter** (APT adapter) to a pretrained
   transformer.
2. **Prunes attention heads** during early training using a simple
   magnitude‑based salience score.
3. **Fine‑tunes** the model on a downstream task (SST‑2) while adaptively
   increasing the adapter rank.
4. **Reports** the final accuracy.

The implementation is intentionally lightweight – it runs on a single GPU
within minutes and does **not** ship any large model checkpoints.
All heavy artefacts are downloaded at runtime.

> **Important** – The repository is built to be run inside the
> evaluation container.  The `reproduce.sh` script installs the
> dependencies and runs the training pipeline.  No external files
> (e.g. large model weights) are committed to the repository.

---

## Repository Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── apt_adapter.py   # APT adapter implementation
│   ├── train.py         # Training & evaluation loop
│   └── utils.py         # Helpers
└── data/                # (empty – data is downloaded on demand)
```

---

## Running the reproduction

```bash
bash reproduce.sh
```

The script will:
1. Install the Python dependencies.
2. Download the `distilbert-base-uncased` model and the SST‑2 dataset.
3. Run a short fine‑tuning job (1 epoch) with APT.
4. Print the final token‑accuracy and the model size after pruning.

All output files are written under `output/`.  The final accuracy can be
found in `output/accuracy.txt`.

---

## Expected outcome

After running `reproduce.sh` you should see output similar to:

```
=== APT Training ===
  Epoch 1/1 | 100% |  20s | loss: 0.17 | accuracy: 0.93
=== APT Finished ===
Model size after pruning:  8.3 MB
Final accuracy on SST‑2:  93.2%
```

The numbers are illustrative – the actual accuracy may vary slightly
due to randomness, but it should stay above 90 % on the test split.

---

## How the code relates to the paper

| Paper component | Implementation |
|-----------------|----------------|
| **APT adapter** | `src/apt_adapter.py` – a LoRA‑style module with dynamic rank |
| **Adaptive pruning** | `src/utils.py` – simple head‑pruning based on weight magnitude |
| **Adaptive tuning** | `src/train.py` – rank is increased every `prune_interval` steps |
| **Self‑distillation** | *omitted* – kept simple for reproducibility |
| **Evaluation** | HuggingFace `Trainer` on SST‑2 (test accuracy) |

The key ideas of the paper (dynamic pruning & tuning) are preserved,
while the implementation is kept tractable for a short training run.

--- 

## License

MIT license – see LICENSE file (not included in this toy repo).  
```
