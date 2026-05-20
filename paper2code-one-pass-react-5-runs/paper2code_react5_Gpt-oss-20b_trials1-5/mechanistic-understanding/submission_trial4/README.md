# Mechanistic Understanding of Alignment Algorithms – Reproduction

This repository contains a minimal, end‑to‑end implementation of the key experiments from
> *A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity*  
> Andrew Lee et al., ICML 2024.

## Overview

The reproduction proceeds in the following steps:

1. **Probe Training** – a linear probe is trained on the Jigsaw toxic comment dataset
   to obtain a *toxic* direction in the hidden state space.
2. **Toxic Vector Extraction** – value vectors from GPT2‑medium that are highly
   aligned with the probe vector are extracted, stacked, and subjected to SVD.
3. **Pairwise Data Generation** – for a set of prompts from Wikitext‑2, we generate
   a *positive* (greedy) and a *negative* (sampling) continuation.
4. **DPO Fine‑tuning** – GPT2‑medium is fine‑tuned using Direct Preference
   Optimization (DPO) on the pairwise dataset.
5. **Evaluation** – we evaluate the unaligned and DPO‑aligned models on:
   * Toxicity (using the `unitary/toxicbert` classifier).
   * Perplexity on Wikitext‑2.
   * F1 on Wikipedia continuations.

All models are pulled from HuggingFace on the fly; no heavy artifacts are stored
in the repository.

## Reproduction Script

Run the following command in a fresh Ubuntu 24.04 container:

```bash
bash reproduce.sh
```

The script will:
* Install dependencies.
* Train the probe and extract toxic vectors.
* Generate the pairwise dataset.
* Fine‑tune GPT2‑medium with DPO.
* Evaluate the models.

Results are printed to the console and saved under the `outputs/` directory.

## Repository Structure

```
├─ config.yaml            # Hyperparameters and paths
├─ reproduce.sh           # Orchestrates the entire pipeline
├─ README.md
├─ requirements.txt
├─ train_probe.py
├─ extract_toxic_vectors.py
├─ pairwise_dataset.py
├─ train_dpo.py
├─ evaluate.py
└─ utils.py
```

## Notes

* The code focuses on **GPT2‑medium** for brevity.  
  The original paper also evaluates Llama2‑7b; adding it would require
  downloading the 7 B checkpoint and may exceed the 1 GB size limit.
* The pairwise dataset generation uses a simple sampling strategy
  instead of full PPLM; this keeps the runtime reasonable while still
  illustrating the DPO dynamics.
* Toxicity evaluation uses the `unitary/toxicbert` model which
  approximates Perspective API scores.

Feel free to adjust `config.yaml` to change batch sizes, epochs,
or to experiment with other models.