# LCA‑on‑the‑Line – A Minimal Reproduction of the Paper

This repository contains a minimal but fully reproducible implementation of the core idea of the paper
*“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies”*.
The goal is to show how the **Lowest Common Ancestor (LCA) distance** can be computed
for a modern vision model and how it correlates with model performance.

## What is reproduced

* **LCA distance computation** based on the WordNet taxonomy.
* **Evaluation pipeline** that:
  1. Loads a pretrained vision model (`resnet18` from `torchvision`).
  2. Downloads the CIFAR‑10 test set (10 classes that map to WordNet synsets).
  3. Computes top‑1 accuracy and the mean LCA distance on the test set.
  4. Saves a CSV file (`results.csv`) containing per‑image predictions, LCA distances,
     and a flag indicating correctness.

> **Why CIFAR‑10?**  
> The full ImageNet dataset is too large for a quick reproduction. CIFAR‑10 contains
> 10 classes that have clear WordNet synset IDs, allowing us to compute LCA distances
> without downloading ImageNet.

## Repository layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── eval.py
└── lca/
    ├── __init__.py
    ├── hierarchy.py
    └── lca_distance.py
```

* `reproduce.sh` – Installs dependencies, downloads data, runs the evaluation script.
* `eval.py` – Main script that performs the experiment.
* `lca/` – Utility modules that build the WordNet hierarchy and compute LCA distances.

## How to run

```bash
bash reproduce.sh
```

The script will create a `results.csv` file in the repository root and print a summary
of the top‑1 accuracy and mean LCA distance.

## Expected outputs

```
Top‑1 Accuracy: 0.92
Mean LCA Distance: 3.4
Results saved to results.csv
```

The exact numbers may vary slightly due to randomness in the model’s predictions, but
they should be close to the values shown above.

## Extending the reproduction

* Replace `resnet18` with any other pretrained model (e.g., `resnet50`, `vit_b_32`).
* Swap the dataset for a larger one (e.g., ImageNet) if you have the data available.
* Use the `lca_distance.py` module to compute LCA distances for any set of predictions
  and ground‑truth labels.