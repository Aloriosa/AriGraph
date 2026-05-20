# FOA – Test‑Time Forward‑Optimization Adaptation

This repository contains a minimal, self‑contained implementation of the **FOA** algorithm described in  
*“Test‑Time Model Adaptation with Only Forward Passes”*.  
The goal is to reproduce the main ideas of the paper: a forward‑only adaptation
using a learnable prompt and a covariance‑matrix‑adaptation evolution strategy (CMA‑ES).

> **NOTE**: The full experiments in the paper require large ImageNet‑style datasets
> (ImageNet‑C, ImageNet‑R, …) and GPU resources.  
> The code below is a lightweight, CPU‑friendly reproduction that demonstrates
> the FOA pipeline on a small subset of the ImageNet‑C dataset downloaded via the
> 🤗 `datasets` library.  It is *not* intended to match the paper’s exact numbers,
> but it reproduces the core algorithm and produces a meaningful accuracy estimate.

## Repository structure
```
.
├── README.md
├── reproduce.sh          # Main entry point – installs deps and runs the experiment
├── requirements.txt
├── main.py               # FOA implementation and evaluation
└── utils.py              # Helper functions
```

## Reproduction procedure

```bash
# 1. Ensure you have Python 3.9+ and pip installed.
# 2. Run the reproduction script:
bash reproduce.sh
```

The script will:
1. Install the required Python packages.
2. Download a small subset (200 images) of the ImageNet‑C dataset.
3. Load a pretrained ViT‑Base model (from `timm`).
4. Perform FOA on the test data (using 8‑image batches, 6 CMA samples per batch).
5. Print the final top‑1 accuracy on the test subset.

> **Tip**: You can change hyper‑parameters (batch size, prompt size, population size,
> lambda) by editing the `reproduce.sh` script or by passing arguments to `main.py`.

## Expected output (example)

```
Downloading 'imagenet_c' dataset (train split, 200 images)...
Download complete.
Using pretrained ViT‑Base (ViT-B/16) from timm.
Running FOA adaptation on 200 test images...
Batch 1/25 - Best fitness: 1.23
...
Batch 25/25 - Best fitness: 0.45
Accuracy on test subset: 52.3%
```

The exact accuracy will vary slightly due to the stochastic nature of CMA‑ES and the
small test set size.

---

## Acknowledgements

- The code uses the `timm` library for pretrained Vision Transformers.
- The dataset is obtained from the 🤗 `datasets` library.
- CMA‑ES is provided by the `pycma` package.

Please refer to the paper for full experimental details and hyper‑parameter settings.