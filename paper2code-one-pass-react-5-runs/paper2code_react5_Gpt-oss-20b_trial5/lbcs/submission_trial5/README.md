# Refined Coreset Selection (LBCS) – Reproduction Repository

This repository contains a lightweight implementation of the
*Refined Coreset Selection* (LBCS) algorithm described in the
ICML 2024 paper by Xiaobo Xia et al.  
The code reproduces the main experimental results on **Fashion‑MNIST,
SVHN, CIFAR‑10** and (optionally) **ImageNet‑1k**.

> Xiaobo Xia, Jiale Liu, Shaokun Zhang, Qingyun Wu, Hongxin Wei,
> Tongliang Liu. **“Refined Coreset Selection: Towards Minimal
> Coreset Size under Model Performance Constraints.”**  
> Proceedings of the 41st International Conference on Machine Learning
> (ICML 2024).

## Features

* **Lexicographic bilevel optimisation** – inner loop trains a neural
  network on a candidate coreset; outer loop performs a pairwise
  lexicographic comparison between two masks.
* **Hard coreset‑size constraint** – the algorithm keeps a target size
  `k` and only accepts a removal if the validation loss stays within
  `ε` of the best loss seen so far.
* **Support for 4 datasets** – Fashion‑MNIST, SVHN, CIFAR‑10 and
  ImageNet‑1k (the latter only if the dataset is available locally).
* **Configurable hyper‑parameters** – all settings are stored in
  `config.json`.  The script `reproduce.sh` will run the experiments
  sequentially for all supported datasets.
* **Lightweight** – only `torch`, `torchvision`, `tqdm`, `numpy`
  are required.  No external binaries or heavy data artifacts are
  shipped with the repository.

## Reproduction

1. **Install system dependencies**

   ```bash
   bash reproduce.sh
   ```

   The script will:
   * update the package list,
   * install Python 3.10, pip and a virtual environment,
   * install the Python dependencies,
   * run the experiments for all datasets,
   * store the results in `output/<dataset>/`.

2. **Inspect the results**

   Each dataset folder contains:
   * `metrics.json` – test accuracy, loss, coreset size, etc.
   * `coreset_indices.npy` – indices of the selected coreset
     (relative to the original training set).

3. **Re‑run with custom settings**

   Edit `config.json` or pass command‑line overrides to `main.py`:

   ```bash
   python src/main.py --dataset cifar10 --epsilon 0.1 --k 5000
   ```

   All hyper‑parameters from the paper (learning rate, batch size,
   number of inner epochs, outer iterations, etc.) are defined in
   `config.json`.  They can be overridden on the command line.

## License

MIT (see `LICENSE`).