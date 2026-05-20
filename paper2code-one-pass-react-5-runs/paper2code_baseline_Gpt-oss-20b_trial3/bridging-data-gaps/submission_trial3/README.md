# DPMs-ANT Reproduction

This repository contains a lightweight, self‑contained reproduction of the
*Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning* paper.
The code implements a simplified version of the proposed
`DPMs-ANT` method, focusing on the key ideas:

1. **Similarity‑guided training** – a learned scalar adapter that shifts the
   diffusion model’s noise prediction.
2. **Adversarial noise selection** – a few steps of gradient ascent on the
   input noise to expose the worst‑case noise for the current model.
3. **Few‑shot fine‑tuning** – training only on 10 target images while keeping
   the pre‑trained network frozen.

> **NOTE**: This implementation is intentionally lightweight to keep the
> repository under 1 GB and to run within the 7‑day Docker runtime. It does
> not aim to reproduce the exact numbers reported in the paper, but it
> demonstrates the core training loop, loss formulation, and generation
> pipeline.

## Repository Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── download_target_images.py
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── adapter.py
│   ├── classifier.py
│   ├── train.py
│   └── generate.py
└── output/
    └── generated_images/
```

* `reproduce.sh` installs dependencies, downloads a small target dataset,
  trains the model, and generates images.
* `config.yaml` holds hyper‑parameters (you can edit it if you wish).
* `src/` contains the core code:
  * `adapter.py` – a tiny learnable scalar per timestep.
  * `classifier.py` – a minimal binary classifier (not used in the final
    training script but kept for completeness).
  * `train.py` – the training loop.
  * `generate.py` – generates 10 images from the fine‑tuned model.

## Reproduction Instructions

```bash
# 1. Run the reproduction script
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`diffusers`, `torch`, `accelerate`, etc.).
2. Download 10 target images from the Internet into `data/target/`.
3. Fine‑tune a pre‑trained DDPM (trained on FFHQ) on these images for 300 steps.
4. Generate 10 images and save them to `output/generated_images/`.

After the script finishes, you will find the generated images and a
`adapter.pt` file containing the learned adapter parameters.

## Expected Outcomes

- **Training**: The script prints a short training log, showing the loss
  decreasing over 300 steps.
- **Generated Images**: 10 PNG images in `output/generated_images/`
  that look like realistic faces (or the target domain, depending on the
  downloaded images). They are not quantitatively evaluated; the goal is
  to demonstrate that the training pipeline works and produces plausible
  samples.

Feel free to modify `config.yaml` to change hyper‑parameters, number of
training steps, learning rate, etc. The code is written to be
easy to adapt for more sophisticated experiments.

Happy experimenting!