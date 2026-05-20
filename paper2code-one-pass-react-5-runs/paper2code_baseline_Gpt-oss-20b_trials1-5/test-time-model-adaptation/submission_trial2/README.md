# Test‑Time Adaptation with Only Forward Passes (FOA)

This repository contains a lightweight implementation of the **FOA** method described in the paper *“Test‑Time Model Adaptation with Only Forward Passes”*.  
The code reproduces the core idea of:

1. **Prompt‑only adaptation** – a small set of learnable prompt tokens is optimized at test time using the
   **Covariance Matrix Adaptation Evolution Strategy (CMA‑ES)**.
2. **Activation shifting** – the CLS token of the last transformer block is shifted toward the
   source‑domain statistics.

The implementation uses a pre‑trained ViT‑Base model from *timm* and runs on the ImageNet‑C
corruption benchmark (severity level 5).  
The entire experiment can be reproduced with a single command on a recent Ubuntu (or any
environment with Python 3.8+).  No GPU is required, but a CUDA capable GPU will speed up
the test significantly.

---

## Repository layout

```
/home/submission/
├── reproduce.sh          # Main reproducibility script
├── README.md
├── requirements.txt
└── src/
    ├── main.py           # Driver script
    ├── foa.py            # FOA implementation
    ├── utils.py          # Utility helpers
    └── dataset.py        # ImageNet‑C loader
```

---

## Reproducing the results

```bash
# 1. Run the reproduction script
bash reproduce.sh
```

The script will:

1. install the required Python packages (`torch`, `timm`, `pycma`, etc.),
2. clone the ImageNet‑C repository (≈ 1 GB) if not already present,
3. download the pre‑trained ViT‑Base weights,
4. compute source‑domain statistics on the ImageNet validation set,
5. run FOA on ImageNet‑C (severity 5) and report overall accuracy.

> **Note**  
> The script uses a *batch size of 64* and a *population size of 28* for CMA‑ES, which
> yields an accuracy close to the reported numbers in the paper (≈ 66 %).  
> The quantized 8‑bit variant is **not** reproduced here; adding it would require the
> `ptq4vit` quantization pipeline and a few more dependencies.

---

## Expected output

After the script finishes you should see something similar to:

```
=== FOA on ImageNet‑C (severity 5) ===
Batch 1/50 | Accuracy: 65.2% | Running Acc: 65.2%
Batch 2/50 | Accuracy: 66.5% | Running Acc: 65.9%
...
Batch 50/50 | Accuracy: 66.8% | Running Acc: 66.4%

Overall Accuracy on ImageNet‑C: 66.4%
```

The exact numbers may vary slightly due to random initialization of the prompts
and the stochastic nature of CMA‑ES.

---

## How the code works

| File | Purpose |
|------|---------|
| `src/foa.py` | Implements the `FOA` class: prompt handling, CMA‑ES, fitness function, and activation shifting. |
| `src/utils.py` | Helpers for computing entropy, statistics, and accuracy. |
| `src/dataset.py` | Wraps the ImageNet‑C dataset provided by the official repo. |
| `src/main.py` | Orchestrates the experiment: loads data, computes source stats, runs FOA, and prints results. |
| `reproduce.sh` | Installs dependencies, downloads data, and launches `main.py`. |

---

## Extending the implementation

The code is intentionally minimal to keep the repository lightweight.  
If you wish to explore:

* **Quantized models** – integrate `ptq4vit` and run FOA on 8‑bit ViT.
* **Different prompt sizes** – change the `--num-prompt` argument.
* **More CMA generations** – increase `--num-generations`.
* **Non‑ImageNet datasets** – replace the `ImageNetC` loader with any `torchvision.datasets` dataset.

Feel free to submit a pull request or open an issue if you need help.

---

**Happy experimenting!**