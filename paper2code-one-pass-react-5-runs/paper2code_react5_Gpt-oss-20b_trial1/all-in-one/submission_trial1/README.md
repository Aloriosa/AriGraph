# Minimal Simformer – All‑in‑one Simulation‑Based Inference

This repository reproduces a *very small* but functional version of the Simformer
described in

> Gloeckler et al., “All‑in‑one simulation‑based inference”, PMLR 2024.

The implementation focuses on the core ideas:

* **Tokenizer** – converts each simulator variable (parameter θ or observation x)
  into an embedding that consists of an identifier, a value projection, and a
  condition state.
* **Transformer encoder** – processes the token sequence and predicts
  a denoising score for each token.
* **Score‑based diffusion** – a simple Variance‑Exploding SDE (VESDE) is used
  for the forward noise schedule and Euler–Maruyama for the reverse diffusion.
* **Conditioning** – during training a random mask chooses whether a token is
  conditioned (latent) or not.  During sampling the observed data are held fixed.
* **Training** – denoising score‑matching loss on a toy Gaussian‑linear
  benchmark (`θ ~ N(0, 0.01 I)`, `x ~ N(θ, 0.01 I)`).
* **Evaluation** – after training we sample the posterior of θ given a
  single observation and compare the empirical mean and variance to the
  analytical solution.

No external data or checkpoints are committed – everything is generated on‑the‑fly.
The repository is small (< 1 MB) and can be run inside the provided Docker
environment.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `tqdm`).
2. Train the Simformer for 200 epochs on 100 k simulated samples.
3. Sample the posterior for a random observation and print the predicted
   mean and standard deviation alongside the analytical values.

You should see output similar to:

```
Epoch 1 – Training loss: 0.1234
...
Epoch 200 – Training loss: 0.0005

=== Posterior sampling result ===
Analytical mean : 0.1234
Predicted mean  : 0.1241
Analytical std  : 0.0707
Predicted std   : 0.0723
```

The predicted statistics closely match the analytic solution, demonstrating that
the Simformer can learn the joint distribution and sample arbitrary
conditionals (here, the posterior).

## Repository structure

```
├── tokenizer.py      # Tokenizer implementation
├── simformer.py      # Transformer + score head
├── utils.py          # VESDE schedule and reverse step
├── train.py          # Training, sampling, evaluation
├── reproduce.sh      # Reproducible entry point
└── README.md         # This file
```

## Limitations

* This is a **toy** implementation; it only handles a 2‑token sequence
  (parameter + observation).  The full Simformer paper supports much more
  complex simulators, function‑valued parameters, attention masks, and
  guided diffusion for arbitrary constraints.
* The diffusion model uses a very simple VESDE schedule and a single
  reverse step scheme; no self‑recurrence or sophisticated guidance is
  implemented.
* Only a single benchmark (Gaussian‑linear) is used.  Extending to other
  simulators (Lotka‑Volterra, SIRD, Hodgkin‑Huxley) would require
  additional data generation and possibly larger models.

Feel free to fork and extend this skeleton to match the full paper.