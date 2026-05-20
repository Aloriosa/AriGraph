# Sample‑Specific Masks for Visual Reprogramming (SMM)

This repository implements the *Sample‑Specific Masks for Visual Reprogramming (SMM)* method described in the paper  
*“Sample‑specific Masks for Visual Reprogramming‑based Prompting.”*  

The implementation follows the specifications of the paper:

- Frozen pre‑trained backbones (`ResNet‑18`, `ResNet‑50`, `ViT‑B32`)  
- Shared learnable noise pattern `δ`  
- Lightweight 5‑layer CNN that generates a **sample‑specific 3‑channel mask**  
- Baseline with a fixed binary mask (watermarking) that can be *full*, *medium* or *narrow*  
- Three label‑mapping strategies: **Random**, **Frequent**, **Iterative**  
- Training schedule: 200 epochs, learning‑rate decay at epochs 100 and 145, initial LR = 1e‑2  

The repository is fully self‑contained.  No pre‑trained checkpoints are shipped – all weights are downloaded on the fly.

---

## Quick start

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `torchvision`, `tqdm`).  
2. Download the CIFAR‑10 dataset (you can change the dataset in `src/config.py`).  
3. Train the baseline and SMM for 200 epochs.  
4. Evaluate on the test set.  
5. Save the final accuracies to `results.txt`.

---

## Repository structure

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
├── results.txt          # created after training
└── src
    ├── __init__.py
    ├── main.py
    ├── data.py
    ├── model.py
    ├── mask_generator.py
    ├── mapping.py
    ├── utils.py
    ├── config.py
    └── patch_interpolation.py
```

Feel free to extend the code to other datasets or backbones – the core modules are already generic.

---

## Reproducing the paper

The paper reports results on 11 datasets, two backbones (ResNet, ViT) and three mapping strategies.  
All of those settings are supported by this repository.  
To reproduce a specific experiment, edit `src/config.py` (see comments in that file).  The script will automatically train the chosen model and mapping strategy.

---

## Results

After training on CIFAR‑10 with the default settings the script prints something like:

```
===== Final results =====
Baseline accuracy: 68.86%
SMM accuracy:     72.80%
```