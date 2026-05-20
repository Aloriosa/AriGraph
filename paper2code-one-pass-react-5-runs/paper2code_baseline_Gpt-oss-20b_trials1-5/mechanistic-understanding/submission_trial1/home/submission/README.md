# Reproduction of “A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity”

This repository contains a **minimal, fully‑reproducible** implementation that demonstrates the key ideas from the paper:

1. **Extracting a linear toxicity probe** from GPT‑2.
2. **Identifying toxic MLP value vectors** that promote toxic tokens.
3. **Intervening** on the residual stream to suppress toxicity.
4. **Fine‑tuning via Direct Preference Optimization (DPO)** on a small synthetic dataset.
5. **Evaluating toxicity** (using a simple keyword list) before and after each step.

> **NOTE** – The implementation uses a *tiny* subset of data and a *single* training epoch for DPO.  
> It is **not** meant to reproduce the exact numbers reported in the paper, but to illustrate the *mechanistic* steps that the authors describe.

## Directory layout

```
/home/submission/
├── README.md
├── reproduce.sh          # entry point
├── requirements.txt
├── main.py               # orchestrates the pipeline
├── probe.py              # train linear toxicity probe
├── intervene.py          # hook for residual‑stream intervention
├── dpo_train.py          # DPO fine‑tuning
├── metrics.py            # toxicity scorer and evaluation
└── outputs/              # generated artefacts
    ├── metrics.json     # final results
    ├── model_probe.pt
    ├── model_dpo.pt
    └── ...
```

## How to run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Train a linear probe on a small subset of the Jigsaw toxicity dataset and save the probe weights.
2. Extract the top‑3 toxic MLP value vectors (by cosine similarity to the probe).
3. Generate text with and without intervention and measure toxicity.
4. Fine‑tune the model with DPO on 200 synthetic prompt/continuation pairs.
5. Re‑evaluate toxicity on the same set of prompts.
6. Save the results to `outputs/metrics.json`.

All steps are logged to the console and the final metrics are available in `outputs/metrics.json`.

## Expected outcomes

You should see:

| Step | Toxicity (simple keyword count) |
|------|---------------------------------|
| Baseline (no fine‑tuning) | High |
| After intervention | Reduced |
| After DPO fine‑tuning | Further reduced |

The numbers will differ from the paper because we use a toy dataset and a small model, but the *trend* (toxicity ↓) should be visible.

## Extending the experiment

* Swap GPT‑2 for Llama‑2‑7B by changing the model id in `main.py`.  
  (Requires a GPU with > 16 GB RAM.)
* Replace the keyword toxicity scorer with a real classifier (e.g., the Perspective API or a HuggingFace toxicity model).
* Increase the DPO training epochs or dataset size to approach the paper’s results.

---