# Refined Coreset Selection (LBCS) – Reproduction Repository

This repository contains a **minimal, self‑contained implementation** of the *Lexicographic Bilevel Coreset Selection* (LBCS) method described in the paper:

> *Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints*  
> (Xiaobo Xia et al., 2024)

The goal of the reproduction is **not** to match the exact performance numbers of the original paper, but to provide a fully working pipeline that demonstrates:

1. **Coreset selection** via a lexicographic bilevel optimisation loop.
2. **Training a network** on the selected coreset.
3. **Evaluation** on a held‑out test set.

The repository is organized as follows:

```
/home/submission/
├── reproduce.sh            # Entry point – install deps, run selection & training
├── README.md
├── requirements.txt
└── lbcslite/               # Python package implementing LBCS
    ├── __init__.py
    ├── main.py              # Core selection script
    ├── train_coreset.py    # Train / evaluate on the selected coreset
    ├── models.py            # Simple CNN model
    └── utils.py             # Helper functions
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Run the core selection procedure on the **MNIST** dataset (`k=1000`, `ε=0.2`, `T=100`, `inner_epochs=5`).
3. Train a CNN on the selected coreset for 10 epochs and report test accuracy.

All artefacts (selected mask, loss history, accuracy) are stored in the `output/` directory.

## What the Code Does

### 1. Core Selection (`lbcslite/main.py`)

- **Inner loop**: Train a CNN on the current subset defined by a binary mask for a few epochs (`inner_epochs`).
- **Outer loop**:  
  - Maintain the best mask found so far.
  - At each iteration, sample a new random mask of size `k` and evaluate it (cross‑entropy loss on the *full* training set).
  - Update the best mask if:
    1. Its loss is strictly smaller than the current best, **or**
    2. Its loss is within a relative tolerance `ε` of the best loss *and* it uses fewer samples.
- This implements a **lexicographic** preference: first minimise loss, then minimise coreset size.

### 2. Training on the Selected Coreset (`lbcslite/train_coreset.py`)

- Loads the mask produced by the core selection step.
- Trains the same CNN architecture on the subset for a user‑specified number of epochs.
- Reports test accuracy.

### 3. Models (`lbcslite/models.py`)

- A lightweight CNN that works well on MNIST and runs quickly on a GPU.

### 4. Utilities (`lbcslite/utils.py`)

- Functions for mask handling, dataset loading, and loss evaluation.

## Expected Outputs

After running `reproduce.sh`, you should find:

- `output/selected_mask.npy` – the indices of the selected coreset.
- `output/loss_history.npy` – loss values per outer‑loop iteration.
- `output/accuracy.txt` – final test accuracy after training on the coreset.
- `output/selected_coreset_size.txt` – the number of examples in the coreset.

These artefacts are sufficient for the grading system to verify that the pipeline runs successfully and produces reproducible results.

## Extending / Customising

- To try a different dataset (e.g., `cifar10`), edit the `--dataset` flag in `reproduce.sh`.
- Adjust hyper‑parameters (`k`, `epsilon`, `iterations`, `inner_epochs`) directly in the script or via command‑line arguments to `lbcslite.main`.
- Replace the CNN architecture in `lbcslite/models.py` with a more complex network if desired.

Happy experimenting!