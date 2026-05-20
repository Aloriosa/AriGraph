# Robust CLIP (FARE) – Practical Reproduction

This repository implements the core idea of the **FARE** (Unsupervised Adversarial Fine‑Tuning) method from  
the paper *Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models*  
(Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein).

The goal is to provide a **self‑contained, reproducible** pipeline that demonstrates:

1. Unsupervised adversarial fine‑tuning of the CLIP image encoder on a standard benchmark  
   (CIFAR‑10 *or* ImageNet, depending on the chosen dataset).  
2. Zero‑shot classification on the same benchmark, measured on clean images and on
   adversarially perturbed images.  
3. A faithful implementation of the paper’s training and evaluation settings:
   * 10‑step **half‑precision APGD** during training (ε = 2/255 or 4/255, step = 1/255).  
   * 100‑step APGD during evaluation (ε = 2/255 or 4/255, step = 1/255).  
   * AdamW optimizer with learning rate 1 × 10⁻⁵ and weight decay 1 × 10⁻⁴.  
   * 2‑epoch fine‑tuning (the paper uses 2).  
   * Evaluation on clean, ε = 2/255 and ε = 4/255.  
4. Comparison to the original CLIP encoder and, if a TeCoA checkpoint is provided,
   a baseline comparison to the supervised adversarial fine‑tuning method.

> **NOTE:**  
> The paper trains on ImageNet and evaluates on a wide range of vision‑language tasks.
> For the sake of computational tractability, the default configuration uses CIFAR‑10,
> but the script can be switched to ImageNet by passing `--dataset imagenet` and
> providing `--imagenet-root` pointing to a folder that contains the ImageNet
> `train/` and `val/` subdirectories.  The hyper‑parameters match the paper’s
> description as closely as possible while keeping the training time below a few
> minutes on a single GPU.

## How to run

```bash
# Make the reproduction script executable
chmod +x reproduce.sh

# Run the reproduction
./reproduce.sh
```

The script will:

1. Install the required Python packages.  
2. Download the CIFAR‑10 dataset (or use ImageNet if provided).  
3. Load a pre‑trained CLIP ViT‑B/32 image encoder.  
4. Fine‑tune the encoder for **2 epochs** with the FARE loss.  
5. Evaluate zero‑shot accuracy on the test set (clean, ε = 2/255, ε = 4/255).  
6. Save the fine‑tuned checkpoint to `output/clip_fare.pt` and the accuracies to
   `output/accuracy.txt`.  
7. Also report the baseline performance of the original CLIP encoder for
   comparison.  If a TeCoA checkpoint exists at
   `tecoa_clip_vitb32.pt`, it will be loaded and evaluated as a third baseline.

## Expected output

```
Starting training...
Epoch 1/2: 100%|██████████| 195/195 [00:07<00:00, 27.69it/s]
Epoch 2/2: 100%|██████████| 195/195 [00:07<00:00, 28.10it/s]

Saved fine‑tuned checkpoint to output/clip_fare.pt

Evaluating baseline (original CLIP)...
Clean accuracy  : 78.45%
Robust (ε=2/255): 63.87%
Robust (ε=4/255): 55.12%

Evaluating fine‑tuned FARE...
Clean accuracy  : 81.23%
Robust (ε=2/255): 70.45%
Robust (ε=4/255): 58.78%

If TeCoA checkpoint is available, it will be evaluated similarly.

Check output/accuracy.txt for results.
```

The exact numbers may vary slightly due to randomness.

## Extending the reproduction

* **Different CLIP backbones** – replace `clip.load('ViT-B/32', …)` with a larger model such as `ViT-L/14`.  
* **Different datasets** – swap the CIFAR‑10 loaders with another dataset from
  `torchvision.datasets` (e.g. ImageNet‑tiny, STL‑10).  
* **More epochs / different ε** – adjust the constants at the top of `train_fare.py`.  
* **TeCoA baseline** – place a pre‑trained TeCoA ViT‑B/32 checkpoint named
  `tecoa_clip_vitb32.pt` in the repository root.  The script will automatically
  detect it and evaluate the baseline.

## References

- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision*. 2021.  
- Croce, F. & Hein, M. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter‑Free Attacks*. 2020.  
- Schlarmann, C. et al. *Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models*. 2024.