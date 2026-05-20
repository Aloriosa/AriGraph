# LCA‑on‑the‑Line Reproduction

This repository implements a minimal, reproducible version of the
*LCA‑on‑the‑Line* benchmark from the paper
> Jia Shi et al., **LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies** (ICML 2024).

The goal is to provide a clean, self‑contained codebase that
* loads a handful of representative models (ResNet18/50, CLIP RN50 & ViT‑B‑32),
* evaluates them on the ImageNet validation set,
* computes the **Lowest Common Ancestor (LCA) distance** using the WordNet hierarchy,
* and prints basic correlation statistics.

> ⚠️ **NOTE**  
> This is a *minimal* reproduction.  
> The full experimental protocol in the paper uses 75 models and 5 OOD datasets
> (ImageNet‑S, ImageNet‑R, ImageNet‑A, ImageNet‑O, etc.).  
> Running the complete benchmark would require > 30 GB of GPU memory
> and several hours of inference time.  
> The current script focuses on the core *implementation* of LCA
> and demonstrates how the metrics are computed.

## Directory layout

```
.
├── data/                 # should contain ImageNet validation images
│   └── imagenet/val/
├── src/
│   ├── __init__.py
│   ├── imagenet_classes.txt   # synset ↔ label mapping (1000 lines)
│   ├── wordnet_utils.py
│   ├── datasets.py
│   ├── models.py
│   ├── evaluator.py
│   └── main.py
├── results/              # output CSV file
├── requirements.txt
├── reproduce.sh
└── README.md
```

## How to run

```bash
# 1. Make sure the ImageNet validation set is available:
#    data/imagenet/val/<synset_id>/<image_files>
#
# 2. Run the reproduction script
./reproduce.sh
```

The script will:

1. Install all dependencies.
2. Download NLTK's WordNet data.
3. Check that the ImageNet validation set is present.
4. Run inference with the four models.
5. Compute ID accuracy, mean LCA distance, and print a correlation between them.
6. Save the metrics to `results/metrics.csv`.

## Extending to OOD datasets

To evaluate on an OOD dataset (e.g. ImageNet‑R):

1. Add a new loader similar to `ImageNetValDataset` in `datasets.py`.
2. Load the OOD data in `main.py` and use the same evaluation functions.
3. Store the OOD accuracy in `ood_acc` and recompute correlations.

## Dependencies

All Python dependencies are listed in `requirements.txt`.  
The script uses `torch`, `torchvision`, `open_clip_torch`, `nltk`, `pandas`, `numpy`, `tqdm`, `scikit‑learn`, and `scipy`.

## License

This code is provided under the MIT license.  The original paper
is © 2024 by the authors; the code here is an independent, educational
implementation.