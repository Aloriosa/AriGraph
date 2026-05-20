# APT (Adaptive Pruning and Tuning) – Lightweight Reproduction

This repository contains a **minimal, runnable** implementation of the key ideas from the paper:

> *APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference*  
> Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao (2024)

The code demonstrates:

1. **LoRA adapters with dynamic rank** – the rank of each adapter can be increased during training.
2. **Structured head pruning** – attention heads with low salience are removed early in training.
3. **Simple training loop** on a GLUE classification task (SST‑2) with a small subset of data for quick demonstration.

> **Important Note**  
> This implementation is *not* a drop‑in replacement for the full paper.  
> The paper uses sophisticated salience scoring, self‑distillation, and extensive hyper‑parameter tuning.  
> Here we provide a lightweight, well‑documented baseline that can be executed on a single A10 GPU within a few minutes.

## Repository Structure

```
├── apt_adapter.py          # Core APT model implementation
├── train_apt.py            # Training script
├── reproduce.sh            # Reproduction script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## How to Run

```bash
# 1. Make the reproducibility script executable
chmod +x reproduce.sh

# 2. Run the script
./reproduce.sh
```

The script will:

1. Create a virtual environment and install the required packages.
2. Train a `bert-base-uncased` model on the SST‑2 task for 3 epochs.
3. Dynamically increase the LoRA rank at epochs 1 and 2.
4. Prune 40 % of the attention heads with the lowest salience.
5. Save the best model and a `results.json` file containing the final accuracy.

The results (final accuracy and best accuracy) will be printed to the console and written to `./results/results.json`.

## Expected Output

After a successful run you should see something like:

```
Epoch 1 – Validation accuracy: 0.9100
Epoch 2 – Validation accuracy: 0.9250
Epoch 3 – Validation accuracy: 0.9310
Final validation accuracy: 0.9310
Reproduction finished. Results are in ./results/results.json
```

The exact numbers vary because of the small random seed and limited data subset,
but the script demonstrates the core APT workflow.

## How the Code Relates to the Paper

| Paper Section | Implementation |
|---------------|----------------|
| **LoRA adapters** (Section 4.1) | `LoRAAdapter` class in `apt_adapter.py` |
| **Dynamic rank** (Section 4.3) | `increase_rank` method & `rank_increase_epochs` argument |
| **Attention‑head pruning** (Section 4.2) | `prune_heads` method (simplified, using mean gradient salience) |
| **Training loop** | `train_apt.py` – uses HuggingFace `Trainer`‑style loop |
| **Reproduction script** | `reproduce.sh` – installs deps and runs training |

> The full paper includes self‑distillation, outlier‑aware salience scoring, and extensive experiments on RoBERTa, T5, and LLaMA.  
> Those components are intentionally omitted here to keep the repo lightweight and fully reproducible within the constraints of the evaluation environment.

## License

This repository is released under the MIT license. The original paper is © 2024.