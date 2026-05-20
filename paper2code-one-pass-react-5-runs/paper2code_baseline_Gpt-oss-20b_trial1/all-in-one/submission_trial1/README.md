# Reproduction of a Toy Two‑Moons Simulation‑Based Inference Experiment

This repository contains a minimal, fully reproducible example that demonstrates the core ideas of the
**All‑in‑one Simulation‑Based Inference** paper (Simformer).  
Instead of implementing the full Simformer architecture, we
train a transformer‑based posterior estimator using the publicly available
`Simulated Bayesian Inference (sbi)` library.  The workflow mimics the
paper’s approach:

1. **Simulator** – A toy “Two Moons” model that generates noisy observations
   from a two‑dimensional parameter vector.
2. **Training** – A transformer neural network is trained to approximate the
   joint distribution \(p(\theta, x)\).
3. **Inference** – Given a new observation, we sample from the posterior
   \(p(\theta \mid x)\) and save the samples to a CSV file.

Although this example is much simpler than the full Simformer, it
illustrates the end‑to‑end pipeline: simulation → training → inference,
and can be extended to the more complex tasks described in the paper
(e.g., Lotka‑Volterra, SIRD, Hodgkin–Huxley) by replacing the simulator
and adjusting the training hyper‑parameters.

## Repository Structure

```
.
├── requirements.txt          # Python packages
├── utils.py                  # Simulator and data generation
├── train.py                  # Training script
├── sample.py                 # Sampling script
├── reproduce.sh              # Reproduction entry point
└── README.md                 # This documentation
```

## Reproduction Instructions

1. **Run the reproduction script**:
   ```bash
   bash reproduce.sh
   ```
   The script installs the required packages, trains the model, and
   generates samples.

2. **Outputs**:
   - `output/posterior_model.pt` – the trained transformer posterior
     model.
   - `output/posterior_samples.csv` – 5 000 posterior samples for a
     randomly generated observation.
   - Console logs detailing the training progress and the test
     observation.

The script is written to be fully self‑contained and to run on a
fresh Ubuntu 24.04 LTS Docker container with an NVIDIA A10 GPU.  It does
not depend on any heavy binary artifacts and keeps the repository
well below the 1 GB limit.

## Extending to the Full Simformer

To reproduce the full results reported in the paper, replace the
`Two Moons` simulator in `utils.py` with the desired simulator
(e.g., Lotka‑Volterra, SIRD, Hodgkin‑Huxley).  Adapt the training
hyper‑parameters accordingly and use the `sbi` library’s
`posterior_estimator` with the `"transformer"` backend to emulate the
Simformer’s transformer‑based score model.

For more advanced diffusion‑based training, the community can integrate
score‑matching objectives and guided diffusion as described in the
paper’s Appendix A3.  The current repository serves as a clean,
readable starting point for such extensions.

## License

This repository is provided under the MIT License.  The code is
intended for educational and research purposes only.