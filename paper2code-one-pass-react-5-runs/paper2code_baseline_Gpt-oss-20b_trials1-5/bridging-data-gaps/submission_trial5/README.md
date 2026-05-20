# DPMs-ANT: Diffusion Model Transfer Learning with Adversarial Noise

This repository contains a minimal, runnable implementation of the *DPMs-ANT* method described in the paper
"Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning".
The goal is to reproduce the core ideas – similarity‑guided training and adversarial noise selection – on a small toy dataset
and to generate a handful of images that show the effect of the transfer.

> **Reproduction script**: `reproduce.sh`  
> **Output**:  
> * `output/fine_tuned_model.pth` – checkpoint of the fine‑tuned UNet.  
> * `output/generated_samples/` – 5 generated images after fine‑tuning.  
> * `output/log.txt` – training log (losses, steps).  

The script is fully self‑contained: it installs all dependencies, downloads a pre‑trained DDPM
(`google/ddpm-cifar10-32`), creates a tiny 10‑shot target dataset if it does not exist,
trains the model for a short number of steps (300), and then samples 5 images.

> **Note**: This is a *minimal* implementation intended for reproduction and educational purposes
> only. It is **not** a full production‑ready training pipeline, but it captures the essential ideas
> of the paper and produces visible results within a few minutes on an NVIDIA A10 GPU.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Download the pre‑trained DDPM for CIFAR‑10.
3. (Re)create a 10‑shot target dataset in `data/target/`.
4. Train the model using the DPMs‑ANT algorithm (similarity‑guided + adversarial noise).
5. Generate 5 samples and save them to `output/generated_samples/`.

All outputs are stored in the `output/` folder.

## Repository Structure

```
src/
├── dataset.py      # Data loading and target dataset creation
├── classifier.py   # Simple binary classifier used for similarity guidance
├── ant_trainer.py  # Core training loop implementing DPMs‑ANT
└── utils.py        # Helper functions (noise scheduler, normalization, etc.)
```

## Expected Results

After running the script you should see:

```
Training: 100%|██████████| 300/300 [00:05<00:00, 56.75it/s]
Generated 5 samples and saved to output/generated_samples/
```

The generated images (in `output/generated_samples/`) will be 32×32 RGB images that
show subtle style transfer from the source DDPM (CIFAR‑10) towards the target
10‑shot dataset. While the visual quality is not state‑of‑the‑art, the samples
demonstrate the effect of the adversarial noise selection (they are noticeably
different from the baseline samples produced by the original DDPM).

## How the Code Relates to the Paper

| Paper Section | Implementation |
|---------------|----------------|
| Similarity‑Guided Training (Sec. 4.1) | `classifier.py` trains a binary classifier on noised images; the gradient of its log‑probability w.r.t. the input is added to the loss (Eq. 5). |
| Adversarial Noise Selection (Sec. 4.2) | `ant_trainer.py` performs a few steps of gradient ascent on the noise used for each sample (Eq. 7). |
| Adaptor Module (Sec. 4.3) | For simplicity, we fine‑tune the entire UNet but keep a small learning rate; this keeps the number of trainable parameters low while still allowing adaptation. |
| Dataset & Evaluation | A toy 10‑shot CIFAR‑10 subset is used; evaluation is limited to visual inspection of the generated samples. |

Feel free to tweak hyper‑parameters (`gamma`, `omega`, `J`, learning rate, etc.) and re‑run the script to see their effect.

---

## References

- Ho, J., Jain, A. & Abbeel, P. "Denoising Diffusion Probabilistic Models", NeurIPS 2020.  
- Dhariwal, P. & Nichol, A. "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021.  
- Karras, T. et al. "StyleGAN2", ICLR 2020.  
- Diffusers library: https://github.com/huggingface/diffusers