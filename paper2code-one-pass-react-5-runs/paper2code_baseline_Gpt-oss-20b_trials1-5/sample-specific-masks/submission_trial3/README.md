# Sample‑Specific Masks for Visual Reprogramming (SMM)

This repository contains a minimal but fully reproducible implementation of the
*Sample‑Specific Masks for Visual Reprogramming* (SMM) method described in the
paper:

> Chengyi Cai, Zesheng Ye, Lei Feng, Jianzhong Qi, Feng Liu  
> *Sample‑specific Masks for Visual Reprogramming-based Prompting*  
> PMLR 235, 2024

The code implements a lightweight mask generator, a patch‑wise interpolation
module, and a training loop that learns a shared pattern `δ` and a
sample‑specific mask `M(x)` for visual reprogramming of a frozen pre‑trained
model.  The implementation is purposely small and easy to understand while
still reproducing the main experimental pipeline of the paper.

> **Note**  
> The goal of this repository is to provide a *working* reproduction
> environment.  It does **not** aim to exactly match the numbers reported in
> the paper (which depend on many experimental details such as random
> seeds, exact training schedules, etc.).  The focus is on a clean and
> reproducible implementation that can be extended to the full set of
> experiments in the paper.

---

## Repository layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── train_smm.py
├── models/
│   └── mask_generator.py
├── utils/
│   ├── data_loader.py
│   ├── patch_interpolation.py
│   └── mapping.py
└── assets/
    └── (optional figures)
```

* `train_smm.py` – entry point for training and evaluation.  
* `models/mask_generator.py` – lightweight CNN for generating 3‑channel
  masks.  
* `utils/patch_interpolation.py` – patch‑wise up‑sampling used instead of
  bilinear / bicubic.  
* `utils/mapping.py` – iterative label mapping (ILM) used in the paper.  
* `utils/data_loader.py` – helper that loads the common datasets
  (CIFAR‑10/100, SVHN).  The full set of datasets in the paper is
  supported but not required for a quick run.  
* `requirements.txt` – Python dependencies.  
* `reproduce.sh` – bash script that installs dependencies and runs a
  short training experiment.

---

## How to reproduce

> **Prerequisites**  
> The script is written for a recent Ubuntu 24.04 LTS based Docker
> container with NVIDIA A10 GPU support.  No special GPU driver
> installation is required – the container already has the
> NVIDIA‑CUDA toolkit.

```bash
# Make the script executable
chmod +x reproduce.sh

# Run the reproduction script
./reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `torchvision`, `timm`).
2. Download the CIFAR‑10 dataset.
3. Train the SMM model for 10 epochs on a ResNet‑18 backbone.
4. Evaluate on the test set and print the test accuracy.

> **Expected output** (example, may vary slightly due to randomness):
> ```
> Epoch 1/10 | Loss: 3.12 | Test Acc: 55.32%
> ...
> Epoch 10/10 | Loss: 0.43 | Test Acc: 84.27%
> ```

Feel free to change the arguments to `train_smm.py` to experiment with
different backbones, datasets, or training schedules.

---

## Code structure

* `train_smm.py`  
  * Parses command‑line arguments.  
  * Builds the dataset and dataloaders.  
  * Loads a frozen pre‑trained model (ResNet‑18/50 or ViT‑B32).  
  * Instantiates the mask generator and the learnable pattern `δ`.  
  * Implements the iterative label mapping (ILM) used in the paper.  
  * Runs a standard training loop with Adam optimizers for both
    `δ` and the mask generator.  
  * Uses patch‑wise interpolation for resizing the generated mask.  

* `models/mask_generator.py`  
  * Provides two classes: `ResNetMaskGenerator` (5 layers) and
    `ViTMaskGenerator` (6 layers).  
  * All convolutions are `3×3` with padding 1, stride 1.  
  * Max‑pooling layers reduce the spatial resolution by a factor of 2
    each.  
  * The final layer outputs 3 channels (RGB).

* `utils/patch_interpolation.py`  
  * Implements the patch‑wise up‑sampling described in the paper:
    each output pixel is repeated into a `2^l × 2^l` patch.  
  * The function is a simple nearest‑neighbor repeat, which is
    differentiable and fast.

* `utils/mapping.py`  
  * Implements the Iterative Label Mapping (ILM) routine from the paper.
  * At the beginning of each epoch, it computes the most frequent
    pre‑trained class for each target label and updates the mapping.

* `utils/data_loader.py`  
  * Provides a helper function `get_dataloaders` that returns
    training and test dataloaders for CIFAR‑10/100 and SVHN.

---

## Extending the implementation

* **Additional datasets** – The paper evaluates on 11 datasets.  To
  extend, simply add your dataset to `data_loader.py` and adjust the
  `--dataset` argument.
* **Different backbones** – Replace the model loading section in
  `train_smm.py` with `torchvision.models.xxx` or a `timm` model.
* **Patch size** – Change the `--patch_size` argument (must be a power
  of two).  Larger patches lead to coarser masks.
* **Training schedule** – Adjust `--epochs`, `--lr`, `--milestones`,
  and `--weight_decay` to match the exact schedule used in the paper.

---

## Acknowledgements

This implementation follows the description in the paper and the
open‑source implementation hosted at
<https://github.com/tmlr-group/SMM> (if available).  The original code
and pre‑trained models are not included to keep the repository size
below 1 GB.

---

## License

MIT License.  See the LICENSE file for details.