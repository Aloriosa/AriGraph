# Reproduction of *Stay on topic with Classifier‑Free Guidance* (CFG)

This repository contains a lightweight, self‑contained reproduction of the core ideas from the paper
"Stay on topic with Classifier‑Free Guidance".  
The implementation focuses on a few representative experiments from the paper:

1. **LAMBADA zero‑shot next‑token prediction**  
2. **Chain‑of‑Thought (CoT) on GSM‑8K**  
3. **Assistant‑style prompting (system + user) with CFG**

All heavy artifacts (model weights, datasets) are downloaded automatically during the first run.
The code uses Hugging‑Face `transformers` and `datasets`.  
The reproduction script (`reproduce.sh`) can be run on a fresh Ubuntu 24.04 LTS Docker container with an
NVIDIA A10 GPU.

## Reproduction Script

```bash
bash reproduce.sh
```

The script will output three result files:

- `lambada_results.txt` – LAMBADA accuracy for γ = 1.0 (baseline) and γ = 1.5 (CFG)
- `gsm8k_results.txt` – GSM‑8K accuracy on a few‑shot CoT prompt for the same two γ values
- `assistant_results.txt` – example outputs for a system‑prompt + user‑prompt pair with and without CFG

## Repository Structure

```
/
├── README.md
├── reproduce.sh
├── cfg_inference.py          # Core CFG logic
├── eval_lambada.py           # LAMBADA evaluation
├── eval_gsm8k.py             # GSM‑8K CoT evaluation
├── eval_assistant.py         # Assistant prompt demo
└── utils.py                  # Helper functions
```

No large binaries are committed – the repository stays well below the 1 GB limit.

## Hyperparameters

The key hyperparameter is the guidance strength `γ`.  
The paper reports that for language‑model CFG the optimal `γ` is usually in the range `[1.0, 2.0]`.  
The default in the scripts is `γ = 1.5`, but you can change it with the `--gamma` flag.

```
python eval_lambada.py --gamma 1.5
python eval_gsm8k.py   --gamma 1.5
```

Feel free to experiment with other values or add more benchmarks following the same pattern.