# APT (Adaptive Pruning and Tuning) – Reproduction

This repository contains a **minimal but functional implementation** of the APT method described in the paper *"APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"*.

The goal is to:

1. **Demonstrate** the core idea of APT – dynamic pruning of model parameters and adaptive tuning via LoRA‑style adapters.
2. **Reproduce a small experiment** (RoBERTa‑base fine‑tuned on SST‑2) that shows how training time and memory can be reduced while keeping accuracy close to full fine‑tuning.

> **NOTE**: This is a **toy implementation**. It captures the essential mechanisms (dynamic masks, salience‑based pruning, adaptive rank increase) but omits many optimisations (efficient pruning schedule, self‑distillation, multiple tasks, LLaMA support, etc.) present in the full paper.

## Reproduction

```bash
bash reproduce.sh
```

The script will:

1. Install the required libraries.
2. Run `python train_apt.py`.
3. Output the final accuracy and training statistics.

## Project Structure

```
├── requirements.txt
├── README.md
├── reproduce.sh
├── train_apt.py
├── models
│   └── apt_adapter.py
└── utils.py
```

### train_apt.py
* Loads the dataset (SST‑2 from GLUE).
* Builds a RoBERTa‑base model and injects `APTLinear` adapters into all linear layers.
* Implements a simple salience‑based pruning schedule.
* Dynamically increases LoRA rank during training.
* Reports accuracy on the dev set.

### models/apt_adapter.py
* Implements `APTLinear`, a linear layer with:
  * Binary input/output masks (`mask_in`, `mask_out`) for pruning.
  * Low‑rank LoRA adaptation (`A`, `B`) with a dynamic rank.
  * Forward pass that respects the masks and adds the LoRA contribution.

## Customisation

* Hyper‑parameters are defined in `train_apt.py`.  
  You can change the number of epochs, batch size, learning rate, target sparsity, etc.
* To experiment on other datasets or models, adapt the dataset loading and model selection sections accordingly.

## Limitations

* The pruning schedule is very simple: after each epoch we prune a fixed fraction of the remaining neurons.
* No self‑distillation or advanced salience metrics are used.
* Only a single task (SST‑2) and a single model (RoBERTa‑base) are supported.

Feel free to extend this repository to include more sophisticated components as described in the paper.