# LCA‑on‑the‑Line Reproduction

This repository contains a **minimal** reproducible implementation of the
LCA‑on‑the‑Line framework described in the paper *“LCA‑on‑the‑Line:
Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies”*.

> **Note**: The original paper evaluates 75 deep‑learning models on the
> full ImageNet validation set and five large‑scale OOD datasets.  
> Re‑implementing that full pipeline would require tens of gigabytes of
> data and weeks of GPU time.  
> Instead, this repository focuses on the *core algorithm*:
> computing the Lowest Common Ancestor (LCA) distance using a WordNet
> hierarchy and correlating it with synthetic model accuracy.
>
> The script generates synthetic predictions for a handful of
> representative models (ResNet‑50, EfficientNet‑B0, CLIP‑RN50).  The
> numbers are **not** meant to match the paper’s results; they only
> demonstrate that the pipeline can be executed end‑to‑end.

## Repository layout

```
├── generated_reproduction.txt   # pip requirements
├── reproduce.sh                 # script that installs deps and runs evaluation
├── evaluate.py                  # core evaluation script
├── README.md                    # this documentation
└── results/                     # will be created by `reproduce.sh`
    ├── results.csv              # per‑model synthetic metrics
    ├── summary.json             # correlation statistics
    └── summary.txt              # human‑readable report
```

## How to run

The container will automatically run the following command:

```bash
bash reproduce.sh
```

The script performs the following steps:

1. Installs Python and the required packages.
2. Downloads the WordNet corpus (required for LCA calculations).
3. Generates synthetic datasets, computes synthetic accuracies
   and LCA distances, and writes the results to `results/`.

The final output is a CSV file with the following columns:

| model | id_top1_acc | id_lca | ood_top1_acc | ood_lca |

and a short summary report in both JSON and plain text format.

## Extending the reproduction

*If you want to evaluate real models on real data*:

1. Replace the synthetic data generation in `evaluate.py` with real
   ImageNet validation and the five OOD datasets.
2. Load the actual checkpoints (e.g., `torchvision.models.resnet50(pretrained=True)`).
3. Compute real predictions and replace the synthetic accuracy
   calculation.
4. The rest of the script (LCA computation, correlation, output) can
   stay unchanged.

The current implementation keeps the core logic – LCA distance
calculation, correlation analysis, and result serialization – in a
single, lightweight script that can run on any machine with a
reasonable amount of RAM (< 2 GB) and a recent Python 3.10+.

## License

This repository is released under the MIT License.  It is provided
“as‑is” without any warranty.  Feel free to adapt or extend it as
needed for your own experiments.