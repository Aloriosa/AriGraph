# DPMs‑ANT: A Lightweight Reproduction of  
> Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning

This repository contains a compact, end‑to‑end implementation of the key ideas from the above paper, adapted to a toy setting:

* **Source domain** – CIFAR‑10 (32×32) pre‑trained DDPM (1000‑step scheduler).  
* **Target domain** – 10‑shot SVHN (10 images).  

The implementation demonstrates:

1. **Similarity‑guided training** – a binary classifier that discriminates source vs. target noisy images and provides a gradient cue for the diffusion model.  
2. **Adversarial noise selection** – a min–max optimisation over Gaussian noise for each training step.  
3. **Adapter modules** – a lightweight 1×1‑conv residual block added to the diffusion model; the backbone is frozen.  
4. **Evaluation** – FID (torch‑fidelity) and LPIPS (pre‑trained VGG) computed against the 10‑shot target set.  

> **NOTE**  
> The experiment is intentionally lightweight for reproducibility. It does **not** reproduce the full‑scale results reported in the paper (FFHQ→Sunglasses, etc.) but faithfully implements the algorithmic pipeline.

## How to Run

```bash
# 1) Prepare the environment
bash reproduce.sh

# 2) Run training & evaluation
python train.py
```

The script will download the pre‑trained DDPM from the HuggingFace hub, train the classifier, fine‑tune the diffusion model, and finally evaluate the model on the 10‑shot SVHN set.  
Results (FID, LPIPS, sample images) are written to `results/` and the console.

## File Structure

```
├─ adapter.py          # Lightweight adapter module
├─ classifier.py       # Binary source/target classifier
├─ utils.py            # Helper functions (metrics, data utils)
├─ train.py            # Main training & evaluation script
├─ reproduce.sh        # Install dependencies & run train.py
├─ requirements.txt    # Python package list
└─ README.md
```

## Hyper‑parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `LR` | `5e-5` | Paper’s 5 × 10⁻⁵ for DDPM |
| `BATCH_SIZE` | `10` | 10‑shot target set |
| `STEPS` | `300` | 300 fine‑tuning iterations (≈ 3 000 effective steps) |
| `GAMMA` | `5.0` | Similarity‑guidance scale |
| `OMEGA` | `0.02` | Adversarial noise step size |
| `J` | `10` | Inner‑loop gradient ascent steps |
| `EVAL_SAMPLES` | `1000` | Number of generated samples for FID/LPIPS |

Feel free to tweak these values for your own experiments.

---

**End of README**