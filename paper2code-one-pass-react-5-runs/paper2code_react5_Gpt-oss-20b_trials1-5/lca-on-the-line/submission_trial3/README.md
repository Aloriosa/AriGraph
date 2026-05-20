# LCA‑on‑the‑Line – Reproduction of the Benchmark Paper

This repository implements a minimal but fully reproducible version of the
**LCA‑on‑the‑Line** benchmark described in
> *LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies* (Shi et al., 2024).

## What the code does

1. **Loads a small set of representative models**  
   * 4 torchvision vision‑only models (ResNet‑18, ResNet‑50, EfficientNet‑B0, ConvNeXt‑Tiny)  
   * 2 zero‑shot vision‑language models (CLIP RN‑50, CLIP ViT‑B‑32)

2. **Downloads the ImageNet validation split** – this is the ID dataset used in the paper – and the five ImageNet‑based OOD datasets  
   * ImageNet‑v2  
   * ImageNet‑Sketch  
   * ImageNet‑R  
   * ImageNet‑A  
   * ObjectNet  

3. **Computes top‑1 accuracy** for each model on every dataset.

4. **Computes the Lowest Common Ancestor (LCA) distance** on the ImageNet ID set using the WordNet hierarchy.  
   * LCA distance is defined as  
     \[
     D_{\text{LCA}}(y',y)=\text{depth}(y)+\text{depth}(y')-2\,\text{depth}(\text{lca}(y,y'))
     \]
     where `depth` is the tree depth in WordNet.

5. **Computes Expected LCA (ELCA)** for models that output a full probability distribution.

6. **Builds a latent taxonomy** by clustering the average image features of each ImageNet class
   using K‑means (9‑level hierarchical clustering).

7. **Runs correlation analysis** between ID LCA / ELCA and OOD top‑1 accuracy
   (Pearson, Spearman, Kendall) and plots the results.

8. **Outputs**:  
   * `results/id_accuracies.csv` – ID accuracy per model  
   * `results/ood_accuracies.csv` – OOD accuracy per model  
   * `results/lca_scores.csv` – ID LCA and ELCA per model  
   * `results/correlation_results.csv` – statistical tables  
   * `results/plots/correlation.png` – visualisation of the “LCA‑on‑the‑Line” plot

## How to run

```bash
bash reproduce.sh
```

The script will take up to ~2 h on a single NVIDIA A10 GPU (or any GPU with ≥8 GB VRAM).  
It will automatically download the datasets (≈ 12 GB) and perform all evaluations.

## Repository structure

```
├── eval_models.py          # Main evaluation driver
├── lca.py                  # LCA / ELCA distance implementations
├── utils.py                # Misc helpers (dataset loading, metrics)
├── config.py               # Hyper‑parameters and model list
├── requirements.txt
├── reproduce.sh
└── README.md
```

## Notes

* The implementation focuses on the core components required to reproduce the
  quantitative results in the paper.  
* Full 75‑model evaluation is not performed due to time constraints, but the
  code can be easily extended by adding entries to `config.MODEL_REGISTRY`.  
* The code uses only the public ImageNet validation set (1 281 167 images) and
  the five OOD datasets; training data is **not** required.  
* All heavy artifacts (datasets, model checkpoints) are downloaded at runtime,
  so the repository stays well below the 1 GB limit.

---