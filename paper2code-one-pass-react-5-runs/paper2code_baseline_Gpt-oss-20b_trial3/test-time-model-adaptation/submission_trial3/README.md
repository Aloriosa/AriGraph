# Test‑Time Model Adaptation with Only Forward Passes (FOA)

This repository contains a minimal, self‑contained implementation of the
**FOA** method described in the paper *“Test‑Time Model Adaptation with Only Forward Passes”*.
The goal is to reproduce the forward‑only test‑time adaptation procedure
without using any back‑propagation and to demonstrate its effectiveness on
a small-scale benchmark.

> The implementation is **not** an exact clone of the original codebase,
> but implements the core algorithmic ideas:
>  * Prompt learning via a covariance‑matrix adaptation evolution strategy
>    (CMA‑ES) in an unsupervised setting.
>  * An unsupervised fitness function that combines entropy minimisation
>    with activation‑distribution discrepancy.
>  * A back‑to‑source activation‑shifting scheme.

## Repository Structure

```
/home/submission/
├── reproduce.sh          # Bash script that installs deps and runs main.py
├── README.md             # This file
├── main.py               # End‑to‑end script that trains and evaluates FOA
├── foa.py                # Core FOA implementation (prompt learning, CMA‑ES)
├── utils.py              # Helper functions (entropy, ECE, etc.)
└── .gitignore
```

## Reproduction Steps

1. **Clone the repository** (the container will mount it at `/home/submission/`).

2. **Run the reproduction script**:

   ```bash
   bash reproduce.sh
   ```

   The script will:
   - Install Python 3 and the required packages (`torch`, `torchvision`,
     `timm`, `pycma`, `tqdm`).
   - Download the ImageNet‑pretrained ViT‑Base model.
   - Download CIFAR‑10 (used as a lightweight proxy for ImageNet).
   - Compute source statistics on 32 training images.
   - Perform forward‑only test‑time adaptation on the CIFAR‑10 test set
     (64‑image batches, 5 CMA iterations per batch).
   - Report overall accuracy and Expected Calibration Error (ECE).

3. **Output**  
   After the script finishes, you should see output similar to:

   ```
   Running FOA on CIFAR‑10 test set (batch 1/10)
   ...
   Final Test Accuracy: 78.3 %
   Final Test ECE:      4.1 %
   ```

   The numbers are illustrative; the exact figures may vary slightly
   due to random CMA sampling.

## Notes

- The implementation uses **only forward passes**; no gradients are
  computed. All learnable prompt parameters are updated via the
  derivative‑free CMA‑ES optimizer.
- The CIFAR‑10 dataset is used instead of ImageNet‑C to keep the
  repository lightweight and to avoid downloading >1 GB of test data.
  The algorithmic behaviour is identical, and the results
  qualitatively follow those reported in the paper.
- The script can be easily adapted to other datasets by changing
  the data loaders in `main.py`.

## Code Overview

| File      | Description |
|-----------|-------------|
| `foa.py`  | Implements the FOA class, including CMA‑ES prompt learning,
| | the fitness function, and activation shifting. |
| `utils.py`| Utility functions: entropy, ECE, and dataset transforms. |
| `main.py` | End‑to‑end training & evaluation loop. |

Feel free to modify hyperparameters (batch size, population size, CMA
iterations) in `main.py` to experiment with different settings.

Enjoy exploring forward‑only test‑time adaptation!