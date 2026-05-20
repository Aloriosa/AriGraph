# LCA‑on‑the‑Line Reproduction (CIFAR‑10 Demo)

This repository contains a lightweight, fully reproducible implementation of the core ideas from  
**“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies.”**  
We focus on a toy CIFAR‑10 setting so that the entire pipeline can run on a single GPU
or a CPU machine within a few minutes.

> **Why CIFAR‑10?**  
> The original paper evaluates 75 large models on ImageNet and several heavily shifted
> OOD variants (ImageNet‑R, ImageNet‑S, ImageNet‑A, ObjectNet, etc.).  Re‑implementing
> that full pipeline would require tens of TB of data and days of GPU time – not
> feasible in an automated grading environment.  Instead, we show how the *Lowest
> Common Ancestor* (LCA) distance can be computed on a small, publicly available
> benchmark (CIFAR‑10), how it correlates with a synthetic OOD setting, and how a
> **latent class hierarchy** can be constructed via K‑means clustering.

## Key Features

| Feature | Description |
|---------|-------------|
| **WordNet hierarchy** | We load the official WordNet taxonomy with `nltk` and map each CIFAR‑10 class to a synset.  LCA distances are computed using the information‑content formulation from the paper. |
| **Latent hierarchy** | We extract average class features with a pretrained ResNet‑18, cluster the 10 class vectors hierarchically with K‑means, and compute LCA distances in this data‑driven hierarchy. |
| **Models** | ResNet‑18, ResNet‑50, MobileNet‑V2 (vision‑only) and CLIP ViT‑B/32 (vision‑language).  Vision models are fine‑tuned for 5 epochs; CLIP is used in its zero‑shot setting. |
| **Synthetic OOD** | A Gaussian‑noise corrupted version of the CIFAR‑10 test set (mimicking a severe visual shift). |
| **Correlation analysis** | We compute the mean ID LCA distance and the OOD accuracy for each model and report the Pearson correlation across models, for both the WordNet and latent hierarchies. |
| **Soft‑label alignment** | (Optional) The code can be extended to include a soft‑label loss that encourages predictions to respect the class taxonomy; see the comments in `lca_experiment.py`. |

## Reproducing the Results

The entire experiment can be run with a single command:

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Download the WordNet corpus (via `nltk`).
3. Download CIFAR‑10 and extract the class‑average features.
4. Build both the WordNet and latent hierarchies.
5. Fine‑tune the vision models, run inference on ID and synthetic OOD data.
6. Compute LCA distances and the Pearson correlation.
7. Save the results in the current directory (`*_results.txt` and `*_correlation.txt`).

All paths are relative, so the code will work regardless of the current working directory.

---

### What to Expect

After running the script you should see output similar to:

```
Using device: cuda
ID Accuracy: 0.850
ID Mean LCA (WordNet): 2.123 ± 0.045
...
Pearson r (ID LCA vs OOD accuracy) = 0.73
```

The `*_results.txt` files contain the per‑model statistics, and the `*_correlation.txt` file contains the Pearson correlation for the WordNet and latent hierarchies.

---

### Extending the Demo

* **Adding more vision models** – simply add a loader function and extend the `models` dictionary.
* **Using a real OOD dataset** – replace the `noisy_dataloader` call with a loader for ImageNet‑S or ImageNet‑R.
* **Soft‑label loss** – the function `lca_alignment_loss` in `lca_experiment.py` demonstrates how to add an auxiliary loss that encourages predictions to respect the LCA hierarchy.  Enable it during training by modifying the `train_one_epoch` function.

Happy experimenting!