# Cramming Tokens into a Single Vector – Minimal Reproduction

This repository contains a lightweight, self‑contained implementation that demonstrates the core idea of the paper  
**“Cramming 1568 Tokens into a Single Vector and Back Again”**.  
The goal is **not** to reproduce the full experimental regime of the paper (which requires multi‑GPU, 8 B‑parameter models and several days of compute).  
Instead we provide a toy experiment that

1. **Compresses short texts** (≈ 50–300 tokens) into one or more trainable embedding vectors (`[mem]`) using a frozen causal language model.  
2. **Decodes** the original text from those vectors, reports token‑level accuracy and cross‑entropy.  
3. Computes the **token‑gain** and **information‑gain** metrics defined in the paper.  
4. Computes the **decoding capacity** (the longest sequence length `L` for which the token‑level accuracy exceeds a user‑defined threshold).  

All heavy artifacts (trained vectors, generated text, metrics) are written into `output/` by the reproduction script.

## Repository layout
```
/home/submission/
├── compress.py           # Main training / decoding code (single or multiple texts)
├── compress_multiple.py  # Wrapper that runs the experiment on many texts and vector counts
├── evaluate.py           # Utility that loads the trained [mem] vectors and evaluates them
├── generate_random_texts.py  # Creates random‑word text files
├── reproduce.sh          # Automates the end‑to‑end run
├── requirements.txt      # Python dependencies
├── sample_texts/         # Tiny example texts (5 files)
└── README.md
```

## Reproducing the results

```bash
bash reproduce.sh
```

The script

1. Installs the required packages (`torch`, `transformers`).
2. Generates a few random‑word text files in `random_texts/`.
3. Runs the compression experiment on the sample texts **and** the random‑word texts for the vector counts `[1, 2, 4, 8]`.
4. Saves per‑text metrics in `output/mem_<k>/metrics.json` and aggregated metrics in `output/mem_<k>/summary.json`.

You can inspect the metrics:

```bash
cat output/mem_1/metrics.json
cat output/mem_1/summary.json
```

## What this toy demo shows

* The paper claims that a single vector can encode up to **1568** tokens for large models.  
  In our toy setting, a *small* model (`distilgpt2`, 68 M params) can reliably reconstruct a 50‑token sentence from one vector.  
* The procedure is a direct instantiation of the paper’s *per‑sample optimization* of `[mem]` vectors:
  * The LM is frozen.
  * Only the `[mem]` embeddings are trainable.
  * Cross‑entropy on the teacher‑forcing targets drives learning.
* The added **token‑gain** and **information‑gain** metrics mirror the exact definitions in the paper (not just an approximation).  
* The **decoding capacity** metric lets you see how many tokens a model can recover from a single vector
  given an accuracy threshold (default 0.99).  
* The script can be easily extended to larger checkpoints (e.g., `gpt2`, `gpt2-medium`, `Llama-3.1-1B`) by changing the `--model_name` flag.

### Limitations

* We use a tiny GPT‑2 variant; the compression capacity is orders of magnitude lower than reported in the paper.  
* We do not evaluate scaling with many vectors, random text, or large models (those would require > 8 B parameters).  
* The learning schedule is fixed and not tuned.  

Nevertheless, the same training loop, loss definition, and decoding logic apply to any HuggingFace causal LM, so you can experiment with larger checkpoints if you wish.

---

Happy experimenting! If you run into issues, feel free to open an issue or check the source code below.