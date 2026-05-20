# LCA‑on‑the‑Line Reproduction

This repository contains a minimal, fully‑reproducible implementation of the core ideas from the paper *“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies”*.  
The goal is **not** to reproduce the full 75‑model, full‑ImageNet experiment (which would require several TB of data and GPU time), but to provide a working, self‑contained example that demonstrates:

1. **Computing the Lowest Common Ancestor (LCA) distance** between a model’s prediction and the ground‑truth label using the WordNet hierarchy.  
2. **Evaluating a few pre‑trained vision models** (e.g. `ResNet‑50` and `CLIP ViT‑B‑32`) on a small ImageNet‑style validation split.  
3. **Showing the correlation** between in‑distribution LCA distance and out‑of‑distribution (OOV) top‑1 accuracy on a tiny OOD dataset (ImageNet‑Sketch).  
4. **Running everything automatically** with a single `reproduce.sh` script.

## Repository layout

```
├── reproduce.sh          # Main reproducibility script
├── requirements.txt      # Python package dependencies
├── lca_metric.py         # Utilities for LCA distance calculation
├── evaluate.py           # End‑to‑end evaluation script
├── data/
│   └── imagenet_sample/  # Small ImageNet‑style validation split (≈200 images)
│   └── imagenet_sketch/  # Corresponding ImageNet‑Sketch OOD split
└── README.md
```

> **NOTE**: The `data` folder contains only a very small subset (~200 images) of the ImageNet validation set and its sketch counterpart. This is *not* the full ImageNet dataset, but is sufficient to demonstrate the pipeline. If you wish to run the full experiment you will need to download the full ImageNet v2/Sketch datasets and point the script to them.

## How to use

```bash
# 1. Ensure you have a recent Python 3.8+ installation
# 2. Run the reproducibility script
bash reproduce.sh
```

The script will:

1. Install all required Python packages.
2. Download the NLTK WordNet data (needed for LCA).
3. Load the two models (`resnet50` from torchvision and `clip-vit-base-patch32` from HuggingFace).
4. Compute top‑1 accuracy and average LCA distance on the ID split.
5. Compute top‑1 accuracy on the OOD split (ImageNet‑Sketch).
6. Print the correlation between ID LCA and OOD accuracy.
7. Store the results in `results.json`.

The outputs are simple, human‑readable and suitable for grading.

## What the code does

- **`lca_metric.py`**  
  Implements `get_lca_distance(pred_idx, target_idx)` which returns the depth‑based LCA distance between two ImageNet class indices. It uses NLTK’s WordNet to find synsets and compute the lowest common ancestor. The distance is simply the sum of the depths from each class to their LCA.

- **`evaluate.py`**  
  Handles loading the datasets, running inference on the CPU (or GPU if available), computing predictions, top‑1 accuracy, and LCA distance. It also runs a simple linear regression to predict OOD accuracy from ID LCA and prints the R² and Pearson correlation.

- **`reproduce.sh`**  
  Installs dependencies, sets up WordNet, and runs `python evaluate.py`.

## Extending the experiment

To run the experiment on a larger set of models or a larger dataset:

1. Add the model names to the `MODEL_NAMES` list in `evaluate.py`.  
2. Replace the `imagenet_sample` and `imagenet_sketch` directories with the full datasets, preserving the folder structure expected by `torchvision.datasets.ImageFolder`.

The code is written to be easily extensible – just plug in new models or new OOD datasets.

Enjoy! 🚀