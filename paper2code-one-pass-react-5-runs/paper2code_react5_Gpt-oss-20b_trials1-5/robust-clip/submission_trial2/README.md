# Robust CLIP – FARE Reproduction

This repository contains a minimal but fully reproducible implementation of the
**FARE** (Unsupervised Adversarial Fine‑Tuning) approach from the paper  
**“Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models”**.

The goal is to:

1. **Train** a FARE‑CLIP model on an unlabeled image collection (CIFAR‑10 is used for
   ease of download; the code can be adapted to ImageNet).
2. **Evaluate** the original CLIP model and the fine‑tuned FARE model on
   zero‑shot image classification (CIFAR‑10 validation set).
3. **Attack** a single image with the APGD procedure from the paper and
   demonstrate that the robust model is less affected.

All heavy artefacts (model checkpoints) are downloaded automatically and the
repository itself stays well below 1 GB.

> **Important**  
> The reproducibility script `reproduce.sh` is the entry point.  
> It installs all required packages, downloads the dataset, trains the FARE
> checkpoint, and runs a demo that prints zero‑shot classification
> accuracies and the impact of an adversarial perturbation.

## How to run

```bash
bash reproduce.sh
```

The script will take roughly 10 minutes on a single GPU (or 20 minutes
on CPU).  It will produce a `checkpoints/fare_clip.pt` file, a
`demo/zero_shot_results.json` file with the classification accuracies, and
`demo/adversarial.txt` with the clean/attack similarity scores.

---

### What is implemented

| Component | Implementation |
|-----------|----------------|
| CLIP download | `clip` library (`openai/clip-vit-base-patch32`) |
| Unsup. adversarial fine‑tuning (FARE) | `train_fare.py` – 2 epochs, 10‑step PGD, 1/255 step size, 2/255 ε |
| APGD attack (half‑precision + single‑precision refinement) | `apgd_attack.py` |
| Zero‑shot classification | `demo.py` |
| Full reproduction pipeline | `reproduce.sh` |

> **Note**: The full paper evaluates on many vision‑language tasks
> (captioning, VQA, etc.) and uses ImageNet for training.  Those
> experiments are outside the scope of this minimal reproduction and
> would require >10 GB of model weights and hours of inference time.
> The code below focuses on the core ideas and demonstrates that the
> unsupervised fine‑tuning preserves nominal accuracy while
> improving robustness to adversarial perturbations.

---

### Reproducibility

- All random seeds are fixed (`torch.manual_seed(42)`).
- Training uses the AdamW optimizer with `lr=1e-5`, `weight_decay=1e-4`,
  a cosine learning‑rate schedule, and a batch size of 128.
- The APGD attack uses 100 iterations in float16, followed by a 10‑step
  refinement in float32.
- The demo reports:
  * Zero‑shot accuracy on CIFAR‑10 validation set for both models.
  * Cosine similarity between image and text embeddings before and after
    perturbation (clean vs. adversarial).

---

### Extending

If you wish to train on ImageNet or evaluate on the full set of tasks
presented in the paper, replace the CIFAR‑10 data loader in
`train_fare.py` with `torchvision.datasets.ImageNet` (requires local
dataset) or with the `datasets` library’s ImageNet split.  Adjust the
image size to 224×224 and the batch size accordingly.

Feel free to open issues or pull requests if you want to add the full
evaluation suite.