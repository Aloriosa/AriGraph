# Refined Coreset Selection – Lightweight Reproduction

This repository contains a minimal, self‑contained implementation of the
*Lexicographic Bilevel Coreset Selection* (LBCS) method described in

> Xiaobo Xia, Jiale Liu, Shaokun Zhang, Qingyun Wu, Hongxin Wei, Tongliang Liu.  
> **Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints** (ICML 2024).

The goal is to reproduce the core ideas and obtain a small coreset on
Fashion‑MNIST that achieves a test accuracy comparable to training on the
full data, while keeping the coreset size well below the initial
selection.

## How to run

The repository is intended to be executed in a clean Ubuntu 24.04 LTS
Docker container with an NVIDIA GPU.  
The provided `reproduce.sh` script installs the required Python packages,
runs the algorithm, and writes the final coreset size and test accuracy to
`output/results.txt`.

```bash
bash reproduce.sh
```

The script will automatically:

1. Install the minimal set of Python dependencies (`torch`, `torchvision`,
   `tqdm`).
2. Download the Fashion‑MNIST dataset.
3. Run LBCS with the hyper‑parameters used in the paper (200‑sample
   coreset, ε = 0.2, 200 outer iterations, 3 inner epochs).
4. Output the final coreset size and test accuracy.

> **Note**  
> The implementation focuses on the key algorithmic components:
> * Inner‑loop training of a neural network on a selected coreset.  
> * Outer‑loop mask optimisation with a lexicographic objective
>   (model performance first, coreset size second).  
> * Evaluation of the primary objective on the *full* training set,
>   as defined in the paper.  
> The script does **not** include baseline comparisons or ImageNet‑1k
> experiments – those are omitted for brevity and to keep the repository
> lightweight.

## Reproducibility

Running the script on a machine equipped with an NVIDIA GPU should finish
within a few minutes. The random seed is fixed to `42` for deterministic
results. The final output file `output/results.txt` contains

```text
Final coreset size: <int>
Final test accuracy: <float>%
Final test loss: <float>
```

Feel free to modify the command‑line arguments in `reproduce.sh` or
`lbc_solve.py` to experiment with different dataset sizes, hyper‑parameters,
or other datasets (SVHN, CIFAR‑10). The code is intentionally simple
yet faithful to the paper’s core ideas.

## License

MIT License. Feel free to use and adapt the code for your research.