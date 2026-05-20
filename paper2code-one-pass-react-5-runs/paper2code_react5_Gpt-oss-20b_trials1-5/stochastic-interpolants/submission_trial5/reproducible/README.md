# Reproduction of *Stochastic Interpolants with Data‑Dependent Couplings*

This repository contains a lightweight, fully reproducible implementation of the core ideas
presented in the paper “Stochastic Interpolants with Data‑Dependent Couplings”.
It implements the stochastic interpolant training algorithm (Algorithm 1 in the paper),
a simple U‑Net velocity model, and a probability‑flow ODE sampler.

> **What we achieved**  
> • Implemented the interpolant training loop that learns a velocity field `b_t(x)`
>   by minimizing the quadratic loss (Equation 7).  
> • Provided a U‑Net architecture with time conditioning, following the paper’s
>   design.  
> • Added a sampling script that integrates the learned velocity field via
>   `torchdiffeq` to produce samples from the target distribution.  
> • Added a simple FID evaluation step (requires a real reference set, e.g. ImageNet).

> **What is missing / simplified**  
> • The code uses a toy base density `x0 = x1 + σz` (rather than the more elaborate
>   corrupted‑image base used in the paper).  
> • No class‑conditional or mask‑conditioned variants are implemented.  
> • Evaluation is lightweight and may be slow on large datasets.  
> • Hyper‑parameters are intentionally modest for quick reproducibility
>   (10 epochs, 32‑batch).  
> • The code is written to run on an A10 GPU but will also work on CPU.

## Usage

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies and create necessary folders.
2. Train the model for 10 epochs on the dataset at `./data/imagenet`.
3. Generate 16 samples and save them to `samples/`.
4. Compute a simple FID (requires a reference set in `./data/imagenet`).

> **Note**  
> Replace `./data/imagenet` with the path to your ImageNet (or ImageNet‑style)
> dataset. The dataset should have the standard folder layout
> `root/class_name/*.jpg`.

## File Structure

```
reproducible/
├── config.yaml          # Hyper‑parameters
├── dataset.py           # ImageNet loader
├── model.py             # U‑Net velocity model
├── train.py             # Training script
├── sample.py            # Sampling script
├── evaluate_fid.py      # FID evaluation (optional)
├── utils.py             # Helper functions
├── reproduce.sh         # End‑to‑end reproducibility script
└── README.md
```

## License

MIT License.  See `LICENSE` for details.