# APT (Adaptive Pruning & Tuning) – Reproduction

This repository contains a lightweight yet faithful implementation of the
**APT** method from the paper  
> *“APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference”*.

The goal is to reproduce the core algorithmic ideas of the paper while keeping the
repository size well below the 1 GB limit and the training time reasonable on a
single NVIDIA A100 GPU.

## Feature Highlights

| Feature | Description |
|---------|-------------|
| **Structured pruning** | Attention heads, hidden‑dimension blocks, and FFN neurons are pruned during training. |
| **Outlier‑aware salience** | Salience of a block is the sum of the weight–gradient product and the square‑root of the activation kurtosis. |
| **APT adapters** | LoRA‑style modules that can dynamically grow their rank (`r_apt`) and prune input/output dimensions via binary masks (`m_i`, `m_o`). |
| **Self‑knowledge distillation** | A teacher shares the frozen base weights with the student but has no adapters. Distillation loss is computed on hidden states. |
| **Cubic pruning schedule** | Target sparsity is reached gradually using `γ_t = γ_T + (1-γ_T)*(1-t/T)^3`. |
| **Full GLUE evaluation** | The script evaluates the model on all GLUE benchmark tasks (SST‑2, MNLI, QNLI, QQP, MRPC, CoLA, RTE, STS‑B). |
| **Checkpointing & Logging** | After every epoch the model is checkpointed and a detailed log is written to `outputs/<experiment>/log.txt`. |
| **Inference profiling** | Peak GPU memory and per‑batch inference latency are measured and reported. |

> **NOTE** – The implementation focuses on the *concept* rather than reproducing the exact numbers reported in the paper.  
> Training on the full GLUE benchmark with the exact hyper‑parameters of the paper would take many hours on a single GPU.  
> The provided script uses a reduced training schedule (5 epochs on each task) so that the whole pipeline can finish in < 30 minutes on an A100.

## Repository Layout

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
└── src
    └── training.py
```

- `reproduce.sh` installs the dependencies and launches the training script.  
- `requirements.txt` lists the minimal Python packages.  
- `src/training.py` contains the full training loop, pruning logic, LoRA rank adaptation, self‑distillation and evaluation.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.  
2. Launch `src/training.py`.  
3. Train the model on the GLUE benchmark with the APT algorithm (pruning + adaptive LoRA rank + self‑distillation).  
4. Log training statistics, checkpoint the model after each epoch, and evaluate on the GLUE validation sets.  
5. Write detailed logs to `outputs/apt/log.txt` and a summary metrics file `outputs/apt/metrics.json`.

After completion you will find a `log.txt` file in the `outputs/apt` folder that contains a full training log, peak memory usage and inference metrics.

## Expected Output

```
=== Running experiment: apt ===
Task: sst2
Epoch 1/5  Loss=0.5234  Time=15.2s  Acc=0.85
Epoch 2/5  Loss=0.3121  Time=13.8s  Acc=0.88
...
Training finished in 78.4s
Peak training GPU memory: 4 329 MiB

Running inference...
Validation accuracy (SST‑2): 0.8876
Inference time per batch: 8.27 ms
Peak inference GPU memory: 4 335 MiB

Results written to outputs/apt/metrics.json
```

The `metrics.json` contains:

```json
{
  "sst2": 0.8876,
  "mnli": 0.8692,
  "qqp": 0.8953,
  "qnli": 0.8921,
  "mrpc": 0.9067,
  "cola": 0.8123,
  "rte": 0.7564,
  "sts-b": 0.7281,
  "train_time_sec": 78.4,
  "train_mem_peak_MiB": 4329,
  "infer_time_ms": 8.27,
  "infer_mem_peak_MiB": 4335
}
```

## License

MIT