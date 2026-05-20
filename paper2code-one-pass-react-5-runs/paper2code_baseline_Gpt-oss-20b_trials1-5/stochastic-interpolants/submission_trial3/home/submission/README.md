# Stochastic Interpolants with Data‑Dependent Couplings – Minimal Reproduction

This repository contains a lightweight, fully reproducible implementation of the
*Stochastic Interpolants with Data‑Dependent Couplings* framework on a toy
in‑painting task using the MNIST dataset.  
The code demonstrates all key components described in the paper:

1. **Data‑dependent coupling** – the base distribution is a noisy, masked
   version of the target image.
2. **Stochastic interpolant** – a linear interpolation with
   `α(t)=1-t`, `β(t)=t`, `γ(t)=0`.
3. **Velocity estimation** – learned by minimizing the quadratic loss
   (Algorithm 1 in the paper).
4. **Sampling** – integration of the probability‑flow ODE
   (Algorithm 2 in the paper) using `torchdiffeq`.

The implementation is intentionally small so that it can be trained and
sampled on a single GPU (or CPU) within a few minutes, making it easy to
run the `reproduce.sh` script in a fresh Docker container.

> **Note**  
> The original paper trains on ImageNet and achieves state‑of‑the‑art
> super‑resolution and in‑painting results.  Here we use MNIST to
> illustrate the core ideas; the qualitative behaviour matches the
> formalism, but the quantitative metrics (FID, etc.) are not
> comparable to the paper’s ImageNet experiments.

## Reproduction

The `reproduce.sh` script installs all dependencies and runs the
training and sampling steps automatically.

```bash
bash reproduce.sh
```

After the script finishes you will find:

* `model.pt` – the trained velocity model.
* `samples/` – a folder containing generated images for all test
  examples (PNG format).
* `train.log` – training loss history.

The script prints a short summary of the results, e.g.

```
Training finished – best loss: 0.076
Generated 1000 samples – saved in samples/
```

## Project Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── train_inpainting.py
├── sample_inpainting.py
├── model.py
├── utils.py
├── requirements.txt
└── samples/          # created at runtime
```

## Code Overview

* `model.py` – defines a simple 3‑layer MLP that takes an image and a time
  embedding and outputs a velocity field of the same shape.
* `utils.py` – helper functions for creating masks, generating the
  data‑dependent base samples, and the time embedding.
* `train_inpainting.py` – implements Algorithm 1 from the paper.
* `sample_inpainting.py` – implements Algorithm 2 from the paper.
* `reproduce.sh` – orchestrates the whole pipeline.

Feel free to experiment with the hyper‑parameters or replace the
network with a U‑Net if you wish to scale to larger images.

## License

MIT License – see the `LICENSE` file.