# Cramming Tokens into a Single Vector – Reproduction Code

This repository contains a minimal, end‑to‑end implementation that reproduces the core idea from the paper
*“Cramming 1568 Tokens into a Single Vector and Back Again”* by Kuratov et al.
We demonstrate how a frozen language model can be guided to reconstruct arbitrary text from a small set of
trainable “memory” vectors (`[MEM]` tokens).  

The implementation uses the Hugging Face `transformers` library and a small, freely available model
(`gpt2`).  
All code is contained in this repository; no large artefacts are stored, so the repository size stays
well under 1 GB.

## How it works

1. **Compression** – `compress.py` trains the embeddings of a few special `[MEM]` tokens so that the
   frozen model can predict a target text with zero loss.  
2. **Decoding** – `decode.py` loads the trained `[MEM]` embeddings and lets the model generate the
   text from them.  
3. **Reproduction script** – `reproduce.sh` installs dependencies, runs the compression and decoding
   steps, and prints the reconstructed text and token‑level accuracy.

The scripts are intentionally simple and only target a single example text (`sample_text.txt`).  
They can be extended to batch experiments or larger models by adjusting the command‑line arguments.

## Usage

```bash
bash reproduce.sh
```

The script will:

1. Install `transformers` and `torch`.
2. Run `compress.py` on `sample_text.txt`.
3. Run `decode.py` to generate the text from the learned `[MEM]` vectors.
4. Print the reconstructed text and token‑level accuracy.

All intermediate files are written to `output/`.

## Expected outcome

Running `reproduce.sh` should produce output similar to:

```
Compression finished. Saved mem vectors to output/mem.pt
Decoding finished. Reconstructed text:
The quick brown fox jumps over the lazy dog. It was a sunny day.
Token accuracy: 100.00%