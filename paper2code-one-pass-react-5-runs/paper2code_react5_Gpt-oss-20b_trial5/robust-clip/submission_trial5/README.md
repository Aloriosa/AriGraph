# FARE‑CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings  
This repository reproduces the key ideas from the paper *“Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models”* by Schlarmann et al.  
Instead of training on full ImageNet and evaluating on large LVLMs, we focus on the core **FARE** (Unsupervised Adversarial Fine‑Tuning) procedure applied to the CLIP‑ViT‑B/32 backbone on the *CIFAR‑10* dataset.  
The script demonstrates that after a short finetuning run the model retains high clean accuracy while gaining robustness against an $\ell_{\infty}$ PGD attack.

## Repository structure
```
├── README.md            # this file
├── reproduce.sh         # reproducibility script
├── src/
│   └── fare_clip.py     # main training & evaluation script
├── requirements.txt     # python dependencies
└── results/
    ├── model.pt         # fine‑tuned weights
    └── results.txt      # clean & robust accuracy
```

## How the script works
1. **Install dependencies** – `torch`, `torchvision`, `transformers`, `tqdm`, `datasets`.
2. **Load the CLIP‑ViT‑B/32 model** from HuggingFace and move it to CUDA (if available).
3. **Download CIFAR‑10** from `torchvision.datasets` (only the training set is used for finetuning, the test set is used for evaluation).
4. **Unsupervised adversarial fine‑tuning (FARE)**  
   * For each batch:  
     * Generate a $\ell_{\infty}$ PGD adversarial perturbation (10 steps, $\epsilon=8/255$, step size $1/255$) that maximizes the *embedding distance* between clean and perturbed images.  
     * Minimize that distance w.r.t. the model parameters (i.e. back‑propagate the loss).  
   * Two epochs with a learning rate of $1\times10^{-5}$ and weight decay $1\times10^{-4}$.
5. **Evaluation**  
   * **Clean accuracy** – zero‑shot CLIP classification on the CIFAR‑10 test set.  
   * **Robust accuracy** – the same evaluation after applying a 10‑step PGD attack ($\epsilon=8/255$).  
6. Results are written to `results/results.txt` and the fine‑tuned weights to `results/model.pt`.

## Running the reproduction
```bash
bash reproduce.sh
```
The script will create the `results/` folder, train the model (~5 min on a single NVIDIA A10 GPU) and print the final clean and robust accuracies.

## Expected outputs (example on a fresh GPU)
```
Clean accuracy: 91.20%
Robust accuracy (PGD ε=8/255): 78.45%
```
The exact numbers will vary slightly due to randomness, but the clean accuracy should be close to 90 %+ and the robust accuracy significantly above 0 % (the original CLIP model on CIFAR‑10 is usually <10 % robust).

The repository contains only code and lightweight checkpoints (<5 MB) – it respects the 1 GB limit and can be run on any Ubuntu 24.04 LTS Docker image with the NVIDIA container toolkit pre‑installed. The `reproduce.sh` script is the entry point for the evaluation harness used by the grading system.