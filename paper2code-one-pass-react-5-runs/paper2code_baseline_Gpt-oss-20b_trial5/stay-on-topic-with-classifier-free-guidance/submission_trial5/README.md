# Reproduction of “Stay on topic with Classifier‑Free Guidance”

This repository contains a minimal, reproducible implementation of
Classifier‑Free Guidance (CFG) for language models as described in the
paper *Stay on topic with Classifier‑Free Guidance* (2024).  
The goal is to demonstrate that CFG can be applied to a vanilla
autoregressive transformer (GPT‑2) without any extra training and that
it changes the generated text and the likelihood of a target sentence.

## Repository layout

```
├── README.md                 # This file
├── reproduce.sh              # Reproduction script
├── cfg_generate.py           # Main Python script
└── requirements.txt          # Optional dependencies list
```

## How to run

```bash
bash reproduce.sh
```

The script will:

1.  Install the required Python packages (`torch`, `transformers`,
    `datasets`).
2.  Download the GPT‑2 base model (`gpt2`) and the tokenizer.
3.  For each of three prompts, generate 20 new tokens with three
    CFG strengths (`γ = 1.0` (baseline), `1.5`, `2.0`).
4.  Compute the perplexity of two short target sentences under the
    baseline and under CFG (`γ = 1.5`).
5.  Write the generated texts to `outputs.txt` and the perplexities to
    `perplexities.txt`.

The outputs are small text files that can be inspected to verify that
the script ran successfully.  The script is fully deterministic
(uses a fixed random seed) and does not rely on any hard‑coded
absolute paths.

## Expected outputs

- **`outputs.txt`**  
  Contains the generated continuations for each prompt and CFG
  strength.

- **`perplexities.txt`**  
  Contains the perplexity values for the two target sentences under
  baseline (`γ = 1.0`) and CFG (`γ = 1.5`).

Feel free to experiment with other prompts, longer generation lengths,
or different CFG strengths by editing `cfg_generate.py`.