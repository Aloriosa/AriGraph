# Reproduction of “A Mechanistic Understanding of Alignment Algorithms”

This repository contains a lightweight, fully reproducible implementation of the core ideas from the paper *A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity*.  
The original experiments involve large models (GPT‑2 Medium ≈ 345 M parameters, LLaMA‑2‑7B ≈ 7 B parameters) and a large dataset of 24 k preference pairs.  To keep the repository small (< 1 GB) and runnable on the provided Docker environment (Ubuntu 24.04 LTS, NVIDIA A10 GPU), we provide a **minimal yet faithful** pipeline that demonstrates the key components:

1. **Model loading** – GPT‑2 Medium and LLaMA‑2‑7B are downloaded on‑the‑fly from HuggingFace.  
2. **Toxicity probe** – a linear probe is trained on the residual stream of the last transformer layer using the Jigsaw toxic‑comment dataset.  
3. **Vector extraction** – key and value vectors that are most aligned with the probe are identified, projected into vocabulary space, and inspected.  
4. **Pair‑wise preference data** – for each Wikidata prompt we generate a non‑toxic continuation with greedy decoding and a toxic continuation with a simple toxicity classifier (Unitary/Toxic‑BERT).  
5. **Direct Preference Optimization (DPO)** – the model is fine‑tuned on the generated preference pairs using the exact loss from Rafailov et al. (2023).  
6. **Evaluation** – toxicity is measured with the same toxicity classifier, perplexity is computed on a held‑out Wikitext‑2 split, and we also show an **unalign** experiment that re‑activates toxicity by scaling key vectors.  

All heavy artifacts (large models, full datasets) are **not** stored in the repository; they are downloaded automatically by the reproduction script.  The repository contains only source code, a `requirements.txt` file, and a single `reproduce.sh` script that runs the entire pipeline end‑to‑end.

## How to run

```bash
# Install dependencies
bash reproduce.sh
```

The script will:

1. Install the required Python packages.  
2. Train the toxic‑comment probe.  
3. Extract and analyse toxic vectors.  
4. Fine‑tune the model with DPO.  
5. Evaluate the aligned model.  
6. Perform an un‑alignment experiment and re‑evaluate.  

All outputs are written to `results/`.  The script is fully deterministic (seeded) and will finish in under 2 h on a single NVIDIA A10 GPU.

> **Note** – If you do not have a GPU or want a quick sanity check, the script will automatically fall back to CPU mode, but the training steps will be slow.

## Repository structure

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── probe_toxicity.py
    ├── extract_vectors.py
    ├── analyze_vectors.py
    ├── train_dpo.py
    ├── evaluate.py
    └── unalign.py
```

The code is heavily commented to aid understanding and to serve as a basis for future extensions.

> **Disclaimer** – The reproduction focuses on demonstrating the methodology rather than reproducing the exact numeric results reported in the paper.  The toy dataset and simplified toxicity classifier are sufficient to show the mechanisms discussed in the paper.

```