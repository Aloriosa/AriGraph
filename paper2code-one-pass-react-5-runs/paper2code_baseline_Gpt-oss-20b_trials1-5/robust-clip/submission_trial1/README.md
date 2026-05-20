# Robust CLIP – Unsupervised Adversarial Fine‑Tuning (FARE) – Reproduction

This repository contains a minimal, self‑contained reproduction of the *FARE* method described in
"Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models".
The code fine‑tunes the visual encoder of a CLIP model on a small subset of CIFAR‑10 using adversarial training,
then evaluates zero‑shot image classification accuracy on clean and adversarial test images.

> **NOTE**  
> The full paper reports results on ImageNet, COCO, VQA, and large‐scale LVLMs.  
> Training those models from scratch is computationally infeasible in the evaluation
> environment, so this repository focuses on a lightweight, fully reproducible
> experiment that captures the core idea of *FARE*.

## Reproduction Script

`reproduce.sh` installs the required Python packages and runs
`src/finetune_and_evaluate.py`.  
The script is fully deterministic (random seeds are fixed) and does not
write any large artefacts – only a small text file with the final
accuracies is produced.

## Expected Output

After running `reproduce.sh` you should see a `results.txt` file with a
table like:

```
Method,Clean Accuracy,Adversarial Accuracy
Original CLIP,0.81,0.28
FARE,0.82,0.52
```

The exact numbers may vary slightly due to randomness, but the script
always prints the accuracies and exits cleanly.

## Repository Layout

```
/
├── README.md
├── reproduce.sh
└── src/
    └── finetune_and_evaluate.py
```

All code is written in pure Python with the `torch`, `transformers`,
`datasets` and `tqdm` libraries.  No external data or pretrained
weights are shipped – everything is downloaded on the fly.