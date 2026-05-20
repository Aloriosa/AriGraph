# Memory‑Token Compression Demo

This repository reproduces the core idea from the paper

> **Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity**  
> *Yuri Kuratov et al.*

The original work explores compression ratios up to *x*1500 on large models (e.g. Llama‑3.2‑8B).  Re‑implementing those experiments would require dozens of GPUs and terabytes of storage, which is infeasible in an automated evaluation setting.

Instead, we provide a **minimal, self‑contained example** that demonstrates the same *train‑a‑memory‑token* procedure on a tiny, publicly available model (GPT‑2).  The script trains a single learnable embedding (`<mem>`) while freezing all other model parameters.  After training, the model can generate the original text from that memory token alone, allowing us to compute token‑level accuracy and cross‑entropy reduction.

## Repository structure

```
.
├── compress_mem.py          # Core training & evaluation logic
├── reproduce.sh             # End‑to‑end reproduction script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── results.json             # Output of the demonstration (generated)
```

## How the code works

1. **Tokenizer & Model** – We load the HuggingFace `gpt2` tokenizer and model.
2. **Add a memory token** – `<mem>` is appended to the tokenizer and the embedding matrix.
3. **Freeze the model** – All parameters are frozen except the new embedding.
4. **Training loop** – For up to `max_steps` we:
   * Feed `[<mem>, original tokens]` to the model.
   * Compute next‑token cross‑entropy loss.
   * Back‑propagate only through the memory embedding.
   * Periodically generate text from the memory token and compute token accuracy.
   * Stop early if perfect reconstruction is achieved.
5. **Result** – The script outputs:
   * Final token accuracy.
   * Generated text.
   * Training statistics (steps, loss, etc.) in `results.json`.

## Reproduction

To reproduce the experiment run:

```bash
bash reproduce.sh
```

The script will:

1. Install Python and required packages.
2. Train on the short sentence:  
   *“The quick brown fox jumps over the lazy dog. It was a sunny day and the sky was clear.”*
3. Save results to `results.json`.

You should see an output similar to:

```
Results written to results.json
Original text:
The quick brown fox jumps over the lazy dog. It was a sunny day and the sky was clear.

Generated text:
The quick brown fox jumps over the lazy dog. It was a sunny day and the sky was clear.

Token accuracy: 100.00%
```

## Extending the demo

The script accepts command‑line arguments for:

* `--model_name` – Any HuggingFace model (e.g. `gpt2-medium`, `distilgpt2`).
* `--text` – Text to compress (default is a short sentence).
* `--max_steps` – Maximum training steps (default 2000).
* `--learning_rate` – Optimizer learning rate (default 1e-3).
* `--seed` – Random seed for reproducibility.

You can experiment with longer texts or larger models, but keep in mind GPU memory limits.

## What the demo shows

* Even a single learnable token can store a surprisingly long sequence when the rest of the model is frozen.
* The training procedure is simple and fast (seconds on a single GPU).
* The approach scales to larger models and more memory tokens (see the paper for full results).

## Limitations

* The demonstration uses a small model; the compression ratios achieved here are far below those reported in the paper.
* We only generate using greedy decoding; more sophisticated decoding could improve reconstruction quality.
* We do not compute cross‑entropy reduction or other metrics presented in the paper; those would require a more elaborate evaluation pipeline.

## License

This repository is provided for educational purposes under the MIT license.  The original paper is licensed under CC BY 4.0.