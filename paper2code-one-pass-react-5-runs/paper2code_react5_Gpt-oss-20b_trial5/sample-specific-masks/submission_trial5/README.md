# Sample‑specific Masks for Visual Reprogramming (SMM)

This repository contains a **fully‑reproducible implementation** of the  
*Sample‑specific Masks for Visual Reprogramming‑based Prompting* paper.

## What the code does

* A lightweight CNN (`MaskGenerator`) produces a **sample‑specific three‑channel mask** for every input image.
* A shared learnable noise pattern (`delta`) is multiplied with the mask and added to the image.
* The resulting image is fed into a **frozen ImageNet‑pretrained ResNet‑18** or **ViT‑B32**.
* Target‑class logits are obtained by an **Iterative Label Mapping (ILM)** that is updated every epoch.
* Baseline comparisons are performed with the following fixed masks:
  * **Pad** – no noise added (mask = 0 everywhere).
  * **Full** – noise added everywhere (mask = 1 everywhere).
  * **Medium** – noise added in a central square of size 56×56.
  * **Narrow** – noise added in a central square of size 28×28.
* The method is evaluated on **10 of the 11 datasets** used in the paper (CIFAR‑10/100, SVHN, GTSRB, Flowers‑102, DTD, UCF‑101, Food‑101, Oxford‑IIIT‑Pet, SUN‑397).  
  EuroSAT is omitted because it is not bundled with `torchvision`; it can be added in the same way if desired.
* The training schedule is intentionally short (5 epochs) so that the whole experiment finishes in a few minutes on a single A10 GPU.

## Reproducibility

* `reproduce.sh` installs the required Python packages, runs the training script, and prints the test accuracy for every dataset and baseline.
* All source code is in the `src/` directory.
* The final test accuracies are written to `results.txt` and printed by the script.

## How to run

```bash
bash reproduce.sh
```

The script will produce a `results.txt` file containing the test accuracies of the SMM method as well as the four baselines for each dataset.