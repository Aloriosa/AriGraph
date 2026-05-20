# Reproduction of “A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity”

This repository contains a **minimalizable** implementation of the key experiments from the paper.  
The goal is to demonstrate the main ideas:

1. **Toxicity probe** – a linear classifier trained on GPT‑2 residuals.  
2. **Toxic value vectors** – the MLP value vectors most aligned with the probe.  
3. **In‑generation intervention** – subtracting a toxic vector from the residual stream.  
4. **DPO fine‑tuning** – a simple implementation of Direct Preference Optimization on a synthetic pairwise dataset.  
5. **Evaluation** – toxicity, perplexity, and a toy F1 score on a small held‑out set.

> **Note:**  
> The experiments use only a *tiny* subset of the data and run for a few epochs, so the numbers will differ from the paper.  
> They are meant to be **reproducible** and **executable** in under 7 days on an Ubuntu 24.04 Docker container with an NVIDIA A10 GPU.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies.  
2. Train a probe and extract toxic vectors.  
3. Run an in‑generation intervention and store the outputs.  
4. Fine‑tune GPT‑2 with DPO on a synthetic pairwise dataset.  
5. Evaluate the fine‑tuned model on the real toxicity prompts, compute perplexity on WikiText‑2, and a toy F1 score.

All intermediate and final results are written to the `results/` directory.

## Repository structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
└── src/
    ├── extract_probe.py
    ├── extract_toxic_vectors.py
    ├── intervene_generate.py
    ├── train_dpo.py
    └── evaluate.py
```

## Expected outputs

* `results/probe_vector.npy` – the learned probe vector.  
* `results/toxic_vectors.json` – top‑5 toxic MLP value vectors and their top tokens.  
* `results/intervention_output.json` – generated continuations with and without the intervention.  
* `results/dpo_model/` – the fine‑tuned DPO model.  
* `results/metrics.json` – average toxicity score, perplexity, and toy F1.

Feel free to inspect the JSON files to see the exact numbers produced by this run.

---

**Happy reproducing!**