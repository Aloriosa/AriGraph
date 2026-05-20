# DPMs‑ANT Reproduction

This repository contains a **minimal, fully‑reproducible implementation** of the *“Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning”* paper (DPMs‑ANT).  
The code reproduces the core ideas:

1. **Similarity‑guided training** – a binary classifier is trained to distinguish the *source* (CIFAR‑10) from the *target* (10 images) domain.  
2. **Adversarial noise selection** – a few‑step gradient‑ascent on the noise vector is used to find the “worst‑case” noise for the current model.  
3. **Fine‑tuning** – only the diffusion U‑Net parameters are updated (no extra memory cost).  

The reproduction is carried out on a standard Ubuntu 24.04 LTS Docker image equipped with an NVIDIA A10 GPU.  
The whole pipeline (data download, training, generation) is executed by the single script `reproduce.sh`.

> **NOTE** – The goal of this repository is to demonstrate the algorithmic workflow and not to match the exact quantitative results of the paper (which require large‑scale training and high‑resolution models). The simplified setup uses the tiny “DDPM‑CIFAR10‑32” model from the HuggingFace Diffusers repository and a handful of 10‑image target samples.

## Expected Outputs

After running `bash reproduce.sh` the following artifacts will be created:

| Path | Description |
|------|-------------|
| `output/ckpt.pth` | Fine‑tuned U‑Net checkpoint |
| `output/generated_*.png` | 100 generated images from the fine‑tuned model |
| `output/metrics.txt` | Simple FID and LPIPS scores (optional) |

The `output/generated_*.png` images should exhibit a noticeable shift towards the target domain style (e.g., more “sunglasses”‑like features) compared with the baseline DDPM‑CIFAR10‑32 model.

## How the Code Relates to the Paper

- **`train.py`** implements the training loop described in Algorithm 1:  
  * sampling of `x₀`, random timestep `t`, Gaussian noise `ε`, adversarial noise update (Eq 7), similarity‑guided loss (Eq 5).  
- **`classifier.py`** (inline inside `train.py`) trains a lightweight CNN to approximate the binary classifier `p_φ(y|x_t)` used by the paper.  
- **`generate.py`** performs a diffusion sampling run with the fine‑tuned U‑Net, mirroring the inference procedure of the paper.  
- **`reproduce.sh`** orchestrates the whole pipeline while ensuring a reproducible environment.

## Running the Reproduction

```bash
# Make the script executable
chmod +x reproduce.sh

# Run the full pipeline
bash reproduce.sh
```

The script may take ~30 minutes on an A10 GPU (depends on network speed for downloading the pretrained model).

## Reproducibility

- The script uses fixed random seeds (`torch.manual_seed(42)`) for deterministic behaviour.  
- All heavy artifacts (pretrained weights, generated images) are **not** committed to git – only source code and tiny placeholder assets.  
- The repository size stays well below 1 GB.

---

**END OF README**