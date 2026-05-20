# Semantic Self‑Consistency Reproduction

This repository implements a lightweight reproduction of the *Semantic Self‑Consistency* paper.  
It demonstrates how to:

1.  Generate multiple chain‑of‑thought (CoT) rationales for a set of reasoning
    questions (AQuA‑RAT, SVAMP, StrategyQA).
2.  Embed the full rationale with a fine‑tuned BERT model (SciBERT for math,
    RoBERTa for commonsense).
3.  Weight the responses using two methods:
    * **Centroid Proximity Weighting (CPW)**
    * **Semantic Consensus Weighting (SCW)**
4.  Aggregate the weighted responses and evaluate accuracy against the
    ground‑truth.

The script can be run on a fresh Ubuntu 24.04 container with an NVIDIA A10 GPU.
The output of the script is a CSV file that contains the accuracy of each
method on each dataset.

> **Important**: The reproduction uses the open‑source *Llama 2 7B* model for
> generation.  No private models or external API keys are required.  If you have
> an OpenAI key you can optionally enable GPT‑3.5 / GPT‑4o mini by setting
> `OPENAI_API_KEY` in the environment – the script will fall back to Llama 2.

## Repository Structure

```
.
├── reproduce.sh          # Install deps & run the experiment
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── embed.py
    ├── evaluate.py
    ├── generate.py
    ├── main.py
    ├── parse_answer.py
    └── weight.py
```

## How to Run

```bash
# From the root of the repository
bash reproduce.sh
```

The script will:

1.  Install the required Python packages.
2.  Download the datasets and the required models to the local cache.
3.  Run the experiment and write `results.csv` in the current directory.

The script can take a few hours on a single A10 GPU, depending on the number of
samples per question (`--num-samples`).  By default 10 samples are used.

## Output

`results.csv` contains one row per dataset & method:

| dataset   | method | accuracy (%) |
|-----------|--------|--------------|
| AQuA-RAT  | baseline | 24.8 |
| AQuA-RAT  | CPW      | 24.6 |
| AQuA-RAT  | SCW      | 25.0 |
| ...       | ...      | ... |

Feel free to adjust hyper‑parameters (`--temperature`, `--max-tokens`, etc.) in
`src/config.py`.

## Contact

For questions or bug reports, open an issue or contact
`tim.knappe@algoverse.ai`.