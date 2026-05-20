# Reproduction of "Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

This repository contains a **minimal, self‑contained implementation** that demonstrates the core ideas of the paper:
* **Forgetting of pre‑trained capabilities** (FPC) during fine‑tuning.
* How simple **knowledge‑retention techniques** (Behavioral Cloning, EWC) can mitigate FPC.

The implementation focuses on two toy environments that capture the two main failure modes described in the paper:
1. A **two‑state MDP** that exhibits *state‑coverage gap*.
2. A **1‑D gridworld** (“AppleRetrieval”) that exhibits *imperfect cloning gap*.

Each environment is trained twice:
* **Vanilla fine‑tuning** (no retention).
* **Fine‑tuning + Behavioral Cloning (BC)**.

The scripts produce CSV files with success statistics which can be plotted to observe the forgetting effect.

> **Note**: This repo does **not** implement the full NetHack / Montezuma / Meta‑World experiments.  
> The goal is to provide a runnable, lightweight reproduction that can be executed on a standard Ubuntu 24.04 LTS Docker container with an NVIDIA A10 GPU (GPU is optional for this toy setup).

## Repository structure