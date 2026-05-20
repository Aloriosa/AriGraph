# SEMA‑Like Continual Learning Demo

This repository contains a minimal, self‑contained reproduction of the core ideas in the paper *Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning*.  
It is **not** a full implementation of the original algorithm, but it demonstrates:

1. How to load a frozen Vision Transformer (ViT) backbone.  
2. How to add a small, learnable adapter on top of the backbone.  
3. How to train the adapter in a continual‑learning style on the CIFAR‑100 dataset (10 incremental tasks).  
4. How to evaluate the final model.

The script is intentionally lightweight so that it can be executed inside the automatically created evaluation Docker container without any additional data or GPU‑specific configuration.

## Repository Structure

```
├── requirements.txt            # Python dependencies
├── reproduce.sh                # Entrypoint for the automatic grader
├── src/
│   ├── train.py                # Training & evaluation logic
│   └── dataset.py              # Data loader utilities
└── README.md
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the dependencies from `requirements.txt`.  
2. Download the CIFAR‑100 dataset.  
3. Train an adapter‑augmented ViT on 10 incremental tasks (each containing 10 classes).  
4. Print the per‑task accuracy and the final average accuracy.

The final accuracy is written to `results.txt` in the root directory.  
Feel free to inspect the source code in `src/` to understand the implementation details.

## Notes

* The code uses only CPU by default.  
* If you have a CUDA‑enabled GPU available, the script will automatically use it.  
* The training schedule is intentionally short (5 epochs per task) to keep the runtime below the 7‑day limit.  
* This is a simplified demo; the original paper uses more sophisticated adapters, representation descriptors, and dynamic expansion logic.