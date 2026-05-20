# LCA‑on‑the‑Line Reproduction

This repository contains a minimal, fully executable reproduction of the
*“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies”* paper.

## Highlights

* **LCA metric** – Computes the Lowest Common Ancestor distance between model predictions and ground‑truth labels using the WordNet hierarchy or a latent hierarchy built with hierarchical K‑means clustering.
* **Benchmark** – Evaluates a small set of pretrained vision models (ResNet‑18, ResNet‑50) and a Vision‑Language model (CLIP ViT‑B/32) on ImageNet and several severe OOD variants (Sketch, Render, Adversarial, ObjectNet).  
  The full paper evaluates 75 models, but this repo keeps the example lightweight while preserving the methodology.
* **Reproduction script** – `reproduce.sh` downloads the necessary dataset subsets, installs dependencies, and runs the evaluation pipeline, producing `results.csv` and `correlation.csv`.
* **Extensible** – The code is structured to easily add more models, OOD datasets, or hierarchical construction methods.

## Running the Reproduction

```bash
bash reproduce.sh
```

The script will:

1. Install system and Python dependencies.
2. Download a 1K‑image ImageNet validation subset and the corresponding OOD subsets.
3. Build a latent hierarchy if requested (optional, disabled by default).
4. Evaluate models, compute top‑1/top‑5 accuracy and average LCA distance.
5. Compute Pearson and R² correlation between ID LCA and OOD top‑1 accuracy.
6. Output `results.csv` and `correlation.csv` in the repository root.

## Expected Outputs

- `results.csv`: Model name, ID accuracy, OOD accuracies, and LCA distances.
- `correlation.csv`: Pearson and R² statistics for each OOD dataset.

The results demonstrate that the in‑distribution LCA distance correlates strongly with the out‑of‑distribution performance, reproducing the key finding of the paper on the provided models and datasets.

## Extending the Experiment

* Add more models to `MODEL_REGISTRY` in `compute.py`.
* Include additional OOD splits by adding their download URLs in `dataset_downloads.py`.
* Enable latent hierarchy construction by passing `--latent-hierarchy` to the script.

## License

This repository is released under the MIT License.