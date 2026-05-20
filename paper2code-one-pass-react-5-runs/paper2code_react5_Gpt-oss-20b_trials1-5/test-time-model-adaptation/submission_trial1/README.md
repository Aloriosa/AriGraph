# FOA Reproduction

This repository reproduces the core ideas of the **Test‑Time Model Adaptation with Only Forward Passes (FOA)** paper.  
The implementation follows the experimental protocol described in the paper:

* ImageNet‑C (severity 5) evaluation
* Full‑precision ViT‑Base and **static 8‑bit PTQ** ViT‑Base
* Forward‑only adaptation via a small input prompt optimized with CMA‑ES
* Unsupervised fitness combining prediction entropy and CLS‑token distribution discrepancy
* Activation shifting that aligns CLS tokens to source‑domain statistics

> **Running the reproduction**  
>
> ```bash
> bash reproduce.sh
> ```
>
> The script installs all dependencies, downloads the datasets, builds the ViT model, applies static 8‑bit PTQ (if requested), and runs FOA (and optionally TENT) on ImageNet‑C severity 5.  
>
> **Results**  
>
> The script prints top‑1/top‑5 accuracy and runtime for FOA and the TENT baseline.  
>
> **Notes**  
>
> * Static 8‑bit PTQ is performed by calibrating the model on a small subset of the ImageNet‑1K validation set (32 images).  
> * The prompt size, CMA‑ES population, and hyper‑parameters match those reported in the paper.  
> * The script is designed to run on CPU (for quantized models) or GPU (for full‑precision models).  
> * The code is intentionally lightweight and contains only the core components needed to reproduce the paper’s results.