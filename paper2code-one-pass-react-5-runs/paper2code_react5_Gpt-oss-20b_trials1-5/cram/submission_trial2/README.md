# Reproduction of “Cramming 1568 Tokens into a Single Vector and Back Again”

This repository contains a minimal, fully reproducible implementation of the per‑sample optimisation procedure presented in the paper
[Cramming 1568 Tokens into a Single Vector and Back Again](https://arxiv.org/abs/2403.XXXX).

The code demonstrates how a frozen language model can reconstruct arbitrary text from a set of trainable memory vectors (`[mem]`) by optimising only those vectors.

## Repository structure

```
├── compress.py          # Core demo script
├── reproduce.sh         # Reproducibility script
├── sample.txt           # Small text used for the demo
└── README.md            # This file
```

## How it works

1. **Load a HuggingFace transformer** (default `gpt2`).
2. **Add a handful of special tokens** `[MEM_0]`, `[MEM_1]`, … that are *only* the trainable parts of the model.  
   All other parameters are frozen.
3. **Train the memory embeddings** to minimise next‑token cross‑entropy on the target text.
4. **Evaluate**:
   * Baseline cross‑entropy / token accuracy (no memory).
   * Memory‑conditioned cross‑entropy / token accuracy (teacher‑forced).
   * Token gain, information gain, theoretical capacity bound, and decoding capacity.
5. **Generate** the reconstructed text by greedy decoding from the memory tokens.
6. **Save** the original text, generated text, and a JSON report with all metrics into the `output/` directory.

## Running the demo

```bash
bash reproduce.sh
```

The script installs the required dependencies, creates a small sample text, and runs `compress.py`.  
All artefacts are written to `output/` and a JSON report (`report.json`) is generated.

> **Note** – The demo uses the publicly available `gpt2` model, which is small enough to run on a single A10 GPU in the evaluation container.  
> The same code applies to any HuggingFace decoder‑only transformer (e.g. Llama‑2, Llama‑3, Pythia) – only the `--model_name` argument needs to be changed.

## Metrics reported

| Metric | Description |
|--------|-------------|
| `baseline_loss` | Cross‑entropy loss on the raw text (teacher‑forced). |
| `mem_loss` | Cross‑entropy loss after training the memory vectors. |
| `baseline_acc` | Token accuracy on the raw text. |
| `mem_acc` | Token accuracy after training the memory vectors. |
| `token_gain` | Difference in the number of correctly predicted tokens. |
| `info_gain` | Reduction in cross‑entropy (`baseline_loss - mem_loss`). |
| `theoretical_bound` | `⌊d_model × 16 / log₂|V|⌋` tokens (Eq. (1) of the paper). |
| `decoding_capacity` | Longest prefix length `L` for which the accuracy is above a user‑specified threshold. |