# Reproduction of “A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity”

This repository implements a **light‑weight but faithful** reproduction of the main experimental pipeline described in the paper.  
The goal is to keep the runtime well below the 7‑day limit while still reproducing the key quantitative trends, and to provide a clear, modular implementation that can be extended.

The pipeline consists of the following stages:

1. **Linear probe training** – a single‑layer classifier that predicts toxicity from the *averaged residual stream* of GPT‑2‑medium.  
2. **Toxic vector extraction** – cosine‑similarity search over GPT‑2 MLP value vectors, followed by an SVD basis extraction.  
3. **Pairwise data generation** – a small set of prompt / chosen / rejected pairs created with GPT‑2 and a simple toxicity filter.  
4. **DPO fine‑tuning** – direct preference optimisation using the `trl` library.  
5. **Evaluation** – compute toxicity, perplexity and an F1‑style overlap on real prompts.  
6. **Intervention** – subtract one toxic vector from the MLP weights.  
7. **Un‑alignment** – scale key vectors to reactivate toxicity.

All stages are orchestrated by `reproduce.sh`.  The data and all checkpoints are stored under the `output/` directory.

> **Important**: The scripts use the *real* GPT‑2‑medium model; the only simplification is that we use a very small pairwise dataset (≈ 2 k examples) and a single‑epoch DPO train step.  This keeps the repository lightweight while still demonstrating the mechanisms described in the paper.

---

## Repository layout

```
.
├── README.md
├── reproduce.sh
├── src/
│   ├── train_probe.py
│   ├── extract_toxic_vectors.py
│   ├── generate_pairwise_data.py
│   ├── dpo_finetune.py
│   ├── evaluate.py
│   ├── intervention.py
│   └── unalign.py
└── requirements.txt
```

**`requirements.txt`** lists the Python packages required by the scripts.  They are installed automatically in the virtual environment created by `reproduce.sh`.

---

## Running the reproduction

```bash
bash reproduce.sh
```

The script will:

1. Set up a clean Python virtual environment.  
2. Install the required packages (`transformers`, `datasets`, `trl`, `torch`, etc.).  
3. Run the stages described above in order.  
4. Write all results to the `output/` directory.

---

## Expected outputs

- `output/probe/probe.pt` – the trained linear probe.  
- `output/toxic_vectors.json` – the 128 most toxic MLP value vectors and the SVD basis.  
- `output/pairwise_data.json` – the pairwise dataset used for DPO.  
- `output/dpo_model/` – the GPT‑2‑medium checkpoint fine‑tuned with DPO.  
- `output/eval/eval.json` – the evaluation metrics (toxicity, perplexity, F1).  
- `output/intervention/intervention.json` – metrics after subtracting one toxic vector.  
- `output/unalign/unalign.json` – metrics after re‑activating toxicity.

All numbers are **approximations** of the paper’s tables but follow the same trends.

---

## Extending the pipeline

The scripts are written as stand‑alone modules, so you can replace any component (e.g., use a different toxicity detector or a larger pairwise dataset) without touching the rest of the code.  All data paths are passed as command‑line arguments, making the pipeline fully reproducible.

---

If you have any questions or would like to contribute improvements, feel free to open an issue or pull request.  
Happy reproducing!