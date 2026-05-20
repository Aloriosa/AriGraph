# Reproduction of “Stay on topic with Classifier‑Free Guidance”

This repository contains a minimal, self‑contained implementation of
Classifier‑Free Guidance (CFG) for autoregressive language models
(GPT‑2, Pythia, LLaMA) as described in the paper *“Stay on topic with
Classifier‑Free Guidance”*.

## Structure

```
├── cfg.py               # Core CFG implementation
├── generate.py          # Demo script that shows CFG, negative prompting
├── reproduce.sh         # Convenience script that installs deps and runs the demo
├── requirements.txt     # Python dependencies
└── README.md
```

## How to reproduce

1. **Clone the repository** (or copy the files into a directory).

2. **Run the reproduction script**:

   ```bash
   bash reproduce.sh
   ```

   This script will:
   - Install the required Python packages.
   - Load the default model (`gpt2`) from HuggingFace.
   - Generate several examples using CFG with a guidance weight of 1.5
     and a negative prompt that mimics an assistant‑style instruction.
   - Store the generated examples in `outputs/generated_examples.json`.

3. **Inspect the results**:

   ```bash
   cat outputs/generated_examples.json | jq .
   ```

   The JSON file contains the original prompt and the generated text.

## Extending the Demo

- **Change the model**: modify `MODEL_NAME` in `generate.py`.  
  Options include `"EleutherAI/gpt-neo-2.7B"`, `"meta-llama/Llama-2-7b-hf"`,
  or any model that supports causal language modelling.

- **Adjust CFG parameters**: edit `GAMMA`, `TEMPERATURE`, `TOP_K`, `TOP_P`,
  `MAX_TOKENS` in `generate.py`.

- **Add more prompts**: append to the `PROMPTS` list.

- **Negative prompting**: Replace `NEGATIVE_PROMPT` with your own instruction
  or use `None` to disable it.

## What is implemented

- **CFG re‑weighting** in logit space:
  ```
  new_logits = unconditional_logits + γ * (conditional_logits - unconditional_logits)
  ```
- **Negative prompting** (CFG with a negative prompt `¬c`):
  ```
  new_logits = neg_logits + γ * (cond_logits - neg_logits)
  ```
- **Top‑k / nucleus filtering**.
- **Simple chain‑of‑thought generation** (demo only).

## Limitations

- The script is a **demonstration**; it does not reproduce the full
  experimental pipeline of the paper (benchmarks, metrics, human
  evaluation, stacking with chain‑of‑thought or self‑consistency,
  entropy analysis, etc.).

- The models are loaded on demand; for large models you need a GPU with
  sufficient VRAM.

- Evaluation on the official benchmarks (LAMBADA, ARC, BoolQ, etc.) is
  left as an exercise.  The `cfg.py` API can be used to build such
  evaluation pipelines.

## Acknowledgements

The implementation follows the equations in the paper and uses the
`transformers` library for model loading and tokenisation.