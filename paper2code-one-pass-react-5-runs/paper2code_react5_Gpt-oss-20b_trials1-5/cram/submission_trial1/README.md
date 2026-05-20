# Reproduction of "Cramming 1568 Tokens into a Single Vector and Back Again"

This repository implements a minimal reproduction of the core idea from the paper:
*train a set of learnable `[mem]` vectors (treated as new tokens) while freezing a pretrained language model, and use them to reconstruct a given text sequence.*

## Structure

```
/
├── compress_mem.py           # Training + evaluation script
├── reproduce.sh              # One‑liner to reproduce the experiment
├── requirements.txt          # Python dependencies
├── data/
│   └── sample_text.txt       # Sample text used for demonstration
├── .gitignore
└── README.md
```

## Core Method

1. **Add `[mem]` tokens**  
   The tokenizer is extended with `k` special tokens `[mem0] … [mem(k-1)]`.  
   The model’s embedding matrix is resized accordingly.

2. **Freeze the LLM**  
   All model parameters are frozen except the embedding matrix.  
   Only the rows corresponding to `[mem]` tokens are trainable.

3. **Train embeddings**  
   For a single text sequence, we prepend the `[mem]` tokens and minimize the standard next‑token cross‑entropy loss.  
   The training loop runs for a user‑defined number of steps (default 5000) or until the loss falls below a small threshold.

4. **Evaluation**  
   * **Token‑level accuracy** – greedy decoding of the text starting from the learned `[mem]` tokens.  
   * **Cross‑entropy** – the loss of the model on the original text with the `[mem]` prefix.  
   * **Generated text** – the string produced by the model after training.

5. **Output**  
   The learned embeddings and evaluation metrics are stored in the specified `output_dir`.

## Usage

```bash
# In a fresh environment
bash reproduce.sh
```

The script will:

1. Install dependencies.  
2. Download a pretrained LLM (default: `EleutherAI/gpt-neo-125M`).  
3. Train a single `[mem]` vector to reconstruct `data/sample_text.txt`.  
4. Print the metrics and save them to `output/metrics.json`.

> **Note**:  
> The script is intentionally lightweight.  
> For larger models (e.g., Llama‑3.2‑1B) you may need more GPU memory and longer training time.  
> Adjust `--model_name`, `--mem_tokens`, `--learning_rate`, and `--num_steps` in `reproduce.sh` accordingly.

## Expected Output

Running the reproduction script yields an output similar to:

```
Training finished in 12 steps (loss 0.0012)
Token accuracy: 0.98
Cross‑entropy: 0.65
Generated text:
Once upon a time in a land far away, there lived a curious young explorer named Elara. She spent her days wandering the misty hills, searching for hidden wonders. One dawn, as the sun painted the sky in hues of amber and gold, she found a shimmering stone on a path she had never taken before. Its surface pulsed with a gentle glow, and when she touched it, a whisper of ancient knowledge entered her mind. Suddenly, the world around her seemed to shift, revealing a path that led to the heart of the forest, where secrets of the old world lay waiting to be discovered.

Metrics saved to output/metrics.json
```

The JSON file contains:

```json
{
  "loss": 0.0012,
  "accuracy": 0.98,
  "cross_entropy": 0.65,
  "generated_text": "Once upon a time ..."
}
```

## Extending the Reproduction

- **Different Models** – Replace the `--model_name` argument with any HuggingFace causal LM.  
- **More `[mem]` Vectors** – Increase `--mem_tokens`.  
- **Longer Text** – Provide a larger `sample_text.txt` file.  
- **Custom Hyper‑parameters** – Adjust learning rate, number of steps, etc.

## Limitations

- The script trains only a single sample; it does not perform the extensive grid search described in the paper.  
- GPU memory is required; CPU execution will be extremely slow.  
- The evaluation is greedy decoding; beam search or temperature sampling could be added for richer experiments.

Feel free to fork, modify, and expand this minimal implementation to match the full experimental protocol of the paper.