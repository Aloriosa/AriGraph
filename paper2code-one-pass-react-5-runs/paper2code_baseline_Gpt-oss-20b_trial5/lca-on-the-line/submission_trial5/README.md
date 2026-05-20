# LCA‑on‑the‑Line Reimplementation (Toy Version)

This repository contains a *toy* implementation of the core idea from the paper
“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class
Taxonomies”.  
The original paper evaluates 75 ImageNet‑based models on several large OOD
datasets.  Re‑implementing all those experiments would require many GB of
pre‑trained checkpoints and test data, which is infeasible for an automated
grading environment.

Instead, this repository demonstrates the key concepts:

1. **LCA Distance** – the taxonomic distance between a predicted class and
   the ground‑truth class.
2. **Correlation with Accuracy** – how LCA distance relates to top‑1 accuracy
   on a small held‑out dataset.

The toy experiment uses the CIFAR‑10 dataset (10 classes) and a single
pre‑trained ResNet‑18 model from `torchvision`.  A hand‑crafted taxonomy for
CIFAR‑10 classes is built from WordNet synsets.  The script `reproduce.sh`
downloads the necessary dependencies, runs the experiment, and writes the
results to `results.json`.

> **Note**: This repository is only meant to showcase the LCA calculation
> and its relationship to accuracy, not to reproduce the full set of
> experiments in the paper.

## Repository Structure

```text
.
├── README.md
├── reproduce.sh
├── requirements.txt
├── src
│   ├── lca.py
│   └── main.py
└── results.json   # created after running reproduce.sh
```

## How to Run

```bash
# From the repository root
bash reproduce.sh
```

After execution, `results.json` will contain:

```json
{
  "top1_accuracy": 0.7654,
  "average_lca_distance": 1.32,
  "num_samples": 10000
}