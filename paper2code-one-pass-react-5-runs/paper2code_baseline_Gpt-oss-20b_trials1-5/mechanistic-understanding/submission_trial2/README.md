# Reproduction of “A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity”

This repository contains a minimal, fully‑automated reproduction of the main experiments from the paper.  
The goal is to:

1. **Train a GPT‑2‑medium model with Direct Preference Optimization (DPO)** on a synthetic pairwise preference dataset that contains toxic and non‑toxic continuations.
2. **Evaluate** the fine‑tuned model on:
   - **Toxicity** (using a local toxicity classifier),
   - **Perplexity** on the Wikitext‑2 test set, and
   - **F1** (token overlap) on a small reference set.

The entire pipeline is executed by the shell script `reproduce.sh`.  
All heavy artifacts (trained model checkpoints) are not committed to the repository – they are generated at runtime.

> **Important:**  
> The results produced by this reproduction are *illustrative* and **do not** match the exact numbers reported in the paper. They are meant to demonstrate that the code can run end‑to‑end and produce plausible metrics.

## Requirements

The script automatically installs the required Python packages.  
Make sure you have a recent **Ubuntu 24.04 LTS** Docker container with an NVIDIA GPU (e.g. A10) and the NVIDIA container toolkit.

## How to Run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install dependencies.
2. Train a DPO model on GPT‑2‑medium for 1 epoch (~15‑20 min on an A10 GPU).
3. Evaluate the model and write metrics to `results.json`.

After the script finishes you should see:

```
Training completed.
Evaluation finished.
Results written to results.json
```

### Inspecting the Results

```bash
cat results.json
```

It will contain JSON similar to:

```json
{
  "toxicity": 0.23,
  "perplexity": 22.1,
  "f1": 0.19
}
```

Feel free to tweak training hyper‑parameters, dataset size, or evaluation metrics by editing the corresponding Python scripts.

## Repository Structure

```
├── reproduce.sh          # Main entry point
├── requirements.txt
├── train_dpo.py          # DPO training script
├── evaluate.py           # Evaluation script
├── utils.py              # Shared helpers
└── README.md
```

All code is written in pure Python using the HuggingFace Transformers, Datasets, and TRL libraries. No heavy model checkpoints are stored in the repo, keeping the repository size well below 1 GB.

## License

MIT © 2024