# Semantic Self‑Consistency Reproduction

This repository implements a lightweight reproduction of the *Semantic Self‑Consistency* paper
(“Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting”).
The code follows the same high‑level pipeline:

1. **Generate multiple chain‑of‑thought rationales** for each question.
2. **Embed** the full rationale with a sentence‑Transformer model.
3. **Weight** the final answers using two semantic weighting schemes:
   * Centroid Proximity Weighting (CPW)
   * Semantic Consensus Weighting (SCW)
4. **Select** the answer with the highest weighted score.
5. **Compare** against the baseline self‑consistency (majority vote).

The reproduction runs on a single GPU (or CPU if no GPU is available) and
produces accuracy reports for three popular reasoning datasets:

| Dataset | Source |
|---------|--------|
| AQuA‑RAT | `datasets` library |
| SVAMP | `datasets` library |
| StrategyQA | `datasets` library |

> **Note**  
> The original paper used large closed‑source models (GPT‑3.5, GPT‑4o mini) as well as
> open‑source 7‑B/8‑B models (Llama 2, Llama 3, Mistral).  
> This reproduction uses the open‑source `gpt2` model (345 M parameters) to keep the
> run time and memory footprint reasonable.  
> While the absolute accuracy numbers will differ, the implementation demonstrates
> the semantic weighting pipeline and can be swapped for any other
> transformer model supported by HuggingFace Transformers.

## Reproduction Script

Run the following command from the repository root:

```bash
bash reproduce.sh
```

The script will:

1. Install required packages.
2. Download the datasets.
3. Run `main.py` which performs the full evaluation.
4. Write the results to `results.txt`.

After execution the file `results.txt` will contain a table of accuracies
for each dataset and each method (Self‑Consistency, CPW, SCW).

## Code Structure

```
├── reproduce.sh          # Driver script
├── requirements.txt      # Python dependencies
├── main.py               # Entry point
├── utils.py              # Helper functions (generation, parsing, weighting)
├── evaluation.py         # Accuracy calculation
├── models.py             # Model and tokenizer loader
├── README.md
└── results.txt           # (generated)
```

## Extending the Reproduction

*Replace the model*  
Edit `main.py` or `models.py` to load any other HuggingFace model
(e.g., `meta-llama/Llama-2-7b-chat-hf`).  
Make sure the tokenizer and generation settings are adapted accordingly.

*Change the sample size*  
Use the `--sample_size` argument to generate more or fewer rationales.

*Add more datasets*  
Download any other dataset via the `datasets` library and add an entry in
`main.py`.

## License

This repository is released under the MIT license.  
All third‑party code and models are used under their respective licenses
as listed in the original papers.