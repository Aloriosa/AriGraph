# FOA – Forward‑Optimization Adaptation

This repository contains a faithful implementation of the
*Test‑Time Model Adaptation with Only Forward Passes* paper
(FOA).  The code reproduces the core ideas of the paper while
remaining lightweight enough to run on a standard Ubuntu 24.04
Docker container with a single NVIDIA A10 GPU.

> **Why a toy dataset?**  
> The official experiments use ImageNet‑C, ImageNet‑R,
> ImageNet‑V2 and ImageNet‑Sketch – all of which require
> large amounts of data and compute.  For the purpose of this
> repository we use CIFAR‑10 as a stand‑in to demonstrate
> the algorithm.  The same code can be run on ImageNet‑C by
> replacing the dataset loading logic with the official
> ImageNet‑C folder structure.

## Features

* **Prompt insertion** – a small number of learnable prompt tokens
  are concatenated to the ViT input.
* **CMA‑ES optimisation** – a derivative‑free evolution strategy
  optimises the prompt using an unsupervised fitness that combines
  prediction entropy with CLS‑token statistics discrepancy.
* **Activation shifting** – the final CLS token is shifted towards
  the source statistics, further aligning the OOD samples with the
  training distribution.
* **Quantisation** – optional dynamic 8‑bit quantisation of the
  ViT model to emulate deployment on resource‑constrained devices.
* **No back‑propagation** – the model weights (including the
  prompt) are never updated with gradients.

## Reproduction

The repository is self‑contained: all required packages are listed
in `requirements.txt`.  The script `reproduce.sh` installs a
virtual environment, downloads CIFAR‑10, builds a pre‑trained
ViT‑Base model, computes source statistics, runs FOA on the test
set and writes predictions to `predictions.csv`.  The script
prints the overall accuracy and the peak GPU memory usage.

```bash
# From the repository root
./reproduce.sh
```

The output will be a CSV file `predictions.csv` containing
the predicted class, the ground‑truth class and the image index.
The script also prints the overall accuracy and peak GPU memory
usage in MB.

## Customising the dataset

If you have the ImageNet‑C dataset locally, you can point the
script to it by passing `--dataset-path /path/to/imagenet_c`
when running `foa.py`.  The dataset should follow the standard
ImageNet folder hierarchy (`train/`, `val/`, etc.) with PNG/JPG
images.  In that case the script will automatically use the
ImageNet‑C validation set for evaluation.

## Hyperparameters

The hyperparameters match the values reported in the paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_prompts` | 3 | Number of learnable prompt tokens |
| `popsize` | 28 | CMA‑ES population size |
| `lambda_` | 0.4 | Weight of activation‑discrepancy term |
| `gamma` | 1.0 | Shift step size for activation shifting |
| `batch_size` | 64 | Batch size for online adaptation |
| `quantize` | `--quantize` | Optional 8‑bit dynamic quantisation |

Feel free to experiment with these values if you wish to
replicate or extend the experiments.