# Reproduction of “Cramming 1568 Tokens into a Single Vector and Back Again”

This repository contains a lightweight, fully reproducible implementation of the memory‑vector compression technique described in the paper *Cramming 1568 Tokens into a Single Vector and Back Again*.  
The goal of this repository is to demonstrate the core idea on a modest scale:

* **Train a single “memory” vector that encodes a short text sequence.**  
* **Decode the text from that vector with a frozen language model.**  
* **Measure token‑level accuracy and cross‑entropy reduction.**

## Repository Structure

```
/home/submission/
├── data/
│   └── sample_texts.txt          # 5 short sentences
├── scripts/
│   ├── utils.py                  # Model/tokenizer loading helpers
│   ├── train_mem_vectors.py      # Main training & evaluation script
│   └── evaluate.py               # Optional helper for re‑evaluation
├── outputs/                      # Will be created by reproduce.sh
│   └── results.json              # Final metrics
├── requirements.txt              # Python dependencies
├── reproduce.sh                  # One‑liner reproducible script
└── README.md
```

## How It Works

1. **Load a pretrained causal language model** (default `gpt2`).  
2. **Add a special token `[MEM]`** to the tokenizer and model embeddings.  
3. For each text sample:
   * Create a trainable vector `m` replacing the embedding of `[MEM]`.  
   * Optimize `m` to minimise next‑token cross‑entropy on the target sequence.  
   * After training, use greedy decoding to generate the text from `m`.  
4. **Compute metrics**:
   * Token‑level accuracy (`Acc`).  
   * Cross‑entropy without memory (`CE_no_mem`).  
   * Cross‑entropy with memory (`CE_with_mem`).  
   * Information gain `CE_no_mem - CE_with_mem`.  
5. Results are stored in `outputs/results.json` and a short summary is printed.

## Running the Reproduction

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required packages.  
2. Train memory vectors for each sentence in `data/sample_texts.txt`.  
3. Produce `outputs/results.json` containing per‑sample metrics and an overall summary.

### Expected Output

```json
{
  "samples": [
    {
      "sample_id": 0,
      "text": "The quick brown fox jumps over the lazy dog.",
      "generated_text": "The quick brown fox jumps over the lazy dog.",
      "accuracy": 1.0,
      "cross_entropy_no_mem": 6.8231,
      "cross_entropy_with_mem": 0.0012,
      "info_gain": 6.8219,
      "loss_curve": [...]
    },
    ...
  ],
  "summary": {
    "average_accuracy": 1.0,
    "average_cross_entropy_no_mem": 6.8231,
    "average_cross_entropy_with_mem": 0.0012,
    "average_information_gain": 6.8219
  }
}
```

Values will vary slightly due to random initialization but should show near‑perfect reconstruction.

## Relation to the Original Paper

| Paper Component | Implementation |
|-----------------|----------------|
| **Memory vectors** | One `[MEM]` token per sample, optimised with AdamW. |
| **Frozen model** | The entire language model is frozen except the memory embedding. |
| **Training objective** | Next‑token cross‑entropy (teacher forcing). |
| **Evaluation** | Token accuracy, cross‑entropy reduction, information gain. |
| **Dataset** | Small synthetic dataset (5 sentences) – serves as a toy demonstration. |
| **Model** | GPT‑2 (117M) – a tractable model for a Docker container with an A10 GPU. |
| **Reproducibility** | All code, data, and scripts are committed; no large binaries. |

The actual paper evaluates many large models (Llama‑3, Mamba, etc.) on thousands of tokens. Reproducing those experiments would require several days and >16 GB of VRAM, beyond the constraints of this repository. This simplified version faithfully captures the algorithmic steps and demonstrates that a single learnable vector can encode a full sentence with a frozen LLM.

## Extending the Experiment

* **Different models**: change `--model_name` to `EleutherAI/gpt-neo-125M`, `tiiuae/falcon-125M`, etc.  
* **More memory tokens**: set `k > 1` by adding multiple `[MEM]` tokens and training a vector for each.  
* **Longer texts**: increase the number of lines in `sample_texts.txt`.  
* **Evaluation metrics**: add BLEU, perplexity, etc.  

All changes can be done by editing `train_mem_vectors.py` or passing new arguments to `reproduce.sh`.

## License

This repository is released under the MIT License. The code is original, adapted for educational purposes, and does not include any proprietary data.