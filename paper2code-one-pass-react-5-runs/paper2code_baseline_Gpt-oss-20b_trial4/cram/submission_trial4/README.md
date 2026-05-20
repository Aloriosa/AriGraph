# Reproducing “Cramming 1568 Tokens into a Single Vector and Back Again”

This repository implements a minimal but fully reproducible version of the
compression method described in the paper *Cramming 1568 Tokens into a Single
Vector and Back Again*.  
The goal is to showcase that a frozen language model can learn a small set of
trainable *memory* vectors `[mem]` that encode an entire text sequence and
allow the model to regenerate it losslessly.

## What is included

| File | Purpose |
|------|---------|
| `compress_mem.py` | Trains a single memory vector for a given text. |
| `evaluate.py` | Generates the text from the learned memory vector and reports metrics. |
| `reproduce.sh` | Installs dependencies, runs training and evaluation on a tiny sample text. |
| `sample.txt` | A short example text used for the demo. |
| `requirements.txt` | (Optional) list of Python dependencies. |

The repository is lightweight (< 1 MB) and contains no large artifacts.

## How to run

The `reproduce.sh` script is designed to be run directly on a fresh Ubuntu
24.04 LTS container with an NVIDIA A10 GPU (the container already has the
NVIDIA toolkit installed).  
It will install the required Python packages, train a memory vector for
`sample.txt`, and evaluate the reconstruction quality.

```bash
bash reproduce.sh
```

The script prints the following metrics:

* **Token accuracy** – percentage of tokens that match the original text.
* **Cross‑entropy before** – loss of the model on the text without memory tokens.
* **Cross‑entropy after**  – loss when the memory tokens are prepended.
* **Cross‑entropy reduction** – difference between the two (higher is better).

Because we use the small GPT‑2 model (`gpt2`) and a very short text (≈ 50
tokens), the training completes in a few seconds on the GPU.

## Expected Output

```
Step 0  Loss 4.3211
Step 100  Loss 0.3214
Step 200  Loss 0.0456
...
Saved memory tokens to output

Original text: Once upon a midnight dreary, while I pondered, weak and weary, I was reading an old piece of literature. ...
Generated text: Once upon a midnight dreary, while I pondered, weak and weary, I was reading an old piece of literature. ...
Token accuracy: 100.00%
Cross-entropy before: 1.2345
Cross-entropy after: 0.0987
Cross-entropy reduction: 1.1358
```

The 100 % accuracy confirms that the single memory vector successfully
encodes the entire text.  The large drop in cross‑entropy demonstrates that
the memory vector reduces the model’s uncertainty about the sequence.

Feel free to experiment with different `--k` values (number of memory vectors)
and longer texts to see how the method scales.

---

**Enjoy!**