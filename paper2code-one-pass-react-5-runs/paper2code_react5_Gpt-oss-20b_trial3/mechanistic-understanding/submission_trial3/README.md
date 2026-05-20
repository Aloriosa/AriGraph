# Reproduction of “A Mechanistic Understanding of Alignment Algorithms:
# A Case Study on DPO and Toxicity”

This repository implements the experimental pipeline described in the paper.
It focuses on GPT‑2‑medium and, optionally, LLaMA‑2‑7b.

## Experimental pipeline

1. **Train a linear probe** (`train_probe.py`) on the Jigsaw Toxic Comment
   Classification dataset.
2. **Extract toxic vectors** (`extract_toxic_vectors.py`) – the script now
   extracts all MLP value vectors, keeps the top 16 384 (≈128²)
   by cosine similarity with the probe.
3. **Compute an SVD basis** of toxic vectors (`svd_toxic_vectors.py`).
4. **Generate a pairwise dataset** (`prepare_pairs.py`) using a lightweight
   PPLM‑style procedure that nudges the model toward toxicity.
5. **Fine‑tune GPT‑2 with DPO** (`dpo_train.py`).
6. **Evaluate** (`evaluate.py`) baseline, aligned and re‑aligned models
   on RealToxicityPrompts using Perspective API or a profanity fallback.
7. **Analyse weight shifts** caused by DPO – GPT‑2 (`analysis_shift.py`) and
   LLaMA‑2‑7b (`analysis_shift_llama.py`).
8. **Demonstrate simple re‑alignment attack** (`realign_gpt2.py`).

All scripts are self‑contained and use only source code and public datasets.
The repository does **not** ship any large model checkpoints; they are
downloaded on demand from HuggingFace.

## Running the full reproduction

```bash
bash reproduce.sh
```

The `reproduce.sh` script will install dependencies, run the steps above,
and produce:

| File | Description |
|------|-------------|
| `probe.pt` | Linear probe weights |
| `toxic_vectors.pt` | Top 16 384 toxic value vectors |
| `svd_toxic.pt` | SVD basis (first 10 singular vectors) |
| `pairs.jsonl` | Pairwise dataset |
| `dpo_gpt2/` | DPO‑fine‑tuned GPT‑2‑medium |
| `eval_results.csv` | Toxicity scores for baseline, aligned, re‑aligned |
| `shift_stats.txt` | Cosine similarity & L2 diff for every GPT‑2 parameter |
| `shift_stats_llama.txt` | Same statistics for LLaMA‑2‑7b (if `llama2_dpo/` exists) |
| `realigned_gpt2/` | Re‑aligned GPT‑2‑medium (key‑vector scaling) |

Optional outputs for LLaMA‑2‑7b are produced only if the checkpoints
are available locally or can be downloaded from HuggingFace.

## Dependencies

```
torch>=2.0
transformers>=4.37
datasets>=2.18
trl>=0.5
tqdm
perspectiveapi
numpy
scipy
huggingface_hub
```

All dependencies are listed in `requirements.txt`.

## Notes

* The LLaMA‑2‑7‑b gating analysis (`llm_gating_analysis.py`) and re‑alignment
  (`llm_realign.py`) are optional scripts and require the base and DPO
  checkpoints to be present locally.
* If a Perspective API key is not available, the evaluation falls back
  to a simple profanity list.  For reproducibility, it is recommended to
  set the `PERSPECTIVE_API_KEY` environment variable and install the
  `perspectiveapi` package.
* All scripts use relative paths and will work in a fresh Docker container
  with an NVIDIA GPU (A10 or equivalent).

Enjoy reproducing the results!