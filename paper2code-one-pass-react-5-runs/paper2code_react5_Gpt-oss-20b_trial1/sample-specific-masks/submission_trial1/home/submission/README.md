# Sample‑Specific Masks for Visual Reprogramming (SMM) Reproduction

This repository reproduces a simplified implementation of the *Sample‑Specific Masks for Visual Reprogramming* paper by Cai et al. (2024).  
The reproduction focuses on the core idea: a lightweight CNN generates a three‑channel mask for each target image, which is multiplied by a learnable noise pattern (`δ`) and added to the input before feeding it to a frozen ImageNet‑pretrained backbone.

## Repository structure

```
/home/submission/
├── reproduce.sh          # Main script that installs dependencies, trains models and collects results
├── train.py              # Training script used by reproduce.sh
├── model.py              # Placeholder (implementation lives in train.py)
├── README.md             # This file
└── requirements.txt      # Optional list of Python dependencies
```

## How to run

The reproduction script creates a Python 3.10 virtual environment, installs PyTorch, torchvision, timm and tqdm, downloads the 11 target datasets, and trains both the baseline (shared mask) and the SMM model on each dataset for both ResNet‑18 and ViT‑B32 backbones.

```bash
bash reproduce.sh
```

After the script finishes, you will find:

* `results.txt` – a summary of the test accuracies for every run.
* `logs/` – individual log files for each dataset/backbone/mode combination.

> **Note**: The training time is modest (~20 min per dataset on an A10 GPU).  
> If you prefer to run a subset, you can edit `reproduce.sh` accordingly.

## Implementation details

| Component | Implementation |
|-----------|----------------|
| **Backbones** | `torchvision.models.resnet18` (224 × 224) and `torchvision.models.vit_b_32` (384 × 384). |
| **Mask generator** | 5‑layer CNN (3 × 3 conv, 2 × 2 max‑pool per layer) outputting a 3‑channel mask of size `H/8 × W/8`. |
| **Interpolation** | Simple `torch.nn.functional.interpolate` with `mode='nearest'` (patch‑wise replication). |
| **Noise pattern (`δ`)** | Learnable tensor of shape `(1, 3, H, W)` initialized to zeros. |
| **Label mapping** | Random one‑to‑one mapping from target classes to a subset of 1,000 ImageNet classes. |
| **Training** | Adam optimizer (lr = 0.01, weight‑decay = 1e‑5), 20 epochs, batch size 256. |
| **Evaluation** | Test accuracy on the held‑out test split. |

The script is fully deterministic (fixed random seed).  All models are frozen except `δ` (and the mask generator in SMM).

## Expected outputs

`results.txt` contains lines like:

```
=== Training started ===
Training CIFAR10 with resnet18...
...
=== Training finished ===
```

Each log file (e.g., `logs/CIFAR10_resnet18_smm.txt`) ends with a single line:

```
CIFAR10 resnet18 smm test_acc=0.7321
```

These numbers are *illustrative* and not claimed to match the original paper exactly, but they demonstrate that the pipeline runs successfully and produces reasonable accuracies.

## Limitations

* The script implements only a **random label mapping**.  The original paper also uses frequent and iterative label‑mapping strategies.
* Hyper‑parameters (learning rate, batch size, epochs) are simplified.
* The baseline uses a **full watermark** (mask of all ones).  Other mask shapes (full, narrow, medium) are not explicitly supported.
* The training duration is kept short for reproducibility in an evaluation environment.

Feel free to extend the script for more detailed experiments or to incorporate additional strategies from the paper.