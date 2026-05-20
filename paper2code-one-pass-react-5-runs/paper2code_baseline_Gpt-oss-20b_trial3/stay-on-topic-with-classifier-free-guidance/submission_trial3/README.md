# Classifier‑Free Guidance (CFG) Reproduction

This repository contains a minimal, fully reproducible implementation of
Classifier‑Free Guidance (CFG) for autoregressive language models as
described in the paper *“Stay on topic with Classifier‑Free Guidance”*.
The goal is to show that CFG can be applied to any pre‑trained
transformer model without additional training, and to demonstrate the
effect on generation quality.

## Repository Structure

```
/home/submission/
├── README.md
├── reproduce.sh          # Bash script that installs dependencies and runs the demo
├── cfg.py                # Core CFG implementation
├── run_demo.py           # Demo script that compares CFG vs. baseline
└── requirements.txt      # (Optional) pip freeze of the used packages