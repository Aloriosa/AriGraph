# APT (Adaptive Pruning & Tuning) Reproduction

This repository contains a lightweight yet functional reproduction of the
*APT: Adaptive Pruning and Tuning* method described in the 2024 ICLR paper
by Zhao et al.  
The implementation focuses on the core ideas:

1. **Adaptive Structured Pruning** – heads of the multi‑head attention are
   pruned gradually during training according to an
   outlier‑aware salience score.
2. **Adaptive Parameter‑Efficient Fine‑Tuning** – a LoRA adapter is added
   to each attention and feed‑forward layer.  The adapter rank is
   increased progressively for the most salient layers.
3. **Self‑Knowledge Distillation** – a teacher copy of the model is kept
   frozen and its hidden states guide the student during the early
   training stages.

The script trains on two GLUE tasks (`SST‑2` and `MNLI`) using the
`distilbert-base-uncased` backbone.  Three baselines are run:

* **LoRA** – static LoRA rank, no pruning.
* **Prune** – static pruning (no LoRA).
* **APT** – joint adaptive pruning and adaptive LoRA rank, with
  self‑distillation.

The reproduction script (`reproduce.sh`) installs the required
dependencies, runs the training loop, and prints the final
metrics (accuracy, training time, peak GPU memory, inference latency,
and inference memory).  
All artifacts are generated at runtime; no heavy binaries are committed
to the repository.

> **Limitations**  
> This is a *educational* reproduction and not a full match of the
> 7‑day experiments described in the paper.  The pruning and salience
> computation are simplified, but the overall adaptive behaviour is
> preserved.

## Running the reproduction

```bash
bash reproduce.sh
```

The script will output a table with the results of all three
configurations.  All code, configuration, and logs are stored under
`apt/`.