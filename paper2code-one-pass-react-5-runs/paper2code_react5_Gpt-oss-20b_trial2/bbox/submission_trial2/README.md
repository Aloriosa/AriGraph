# BBox‑Adapter (Reproduction)

This repository contains a lightweight, fully reproducible implementation of the
**BBox‑Adapter** method described in the paper *“BBox‑Adapter: Lightweight Adapting for Black‑Box Large Language Models”*.

The implementation follows the paper closely:
* **Adapter size**: 0.1B–0.3B parameters (full DeBERTa‑base or DeBERTa‑large)  
* **Ranking‑based NCE loss** for distinguishing source vs. target data  
* **Online adaptation**: each epoch evaluates the adapter on the training set, storing the best predictions for use as additional negatives in the next epoch  
* **Adaptive inference**: combines the adapter score with the black‑box LLM’s log‑probability of a candidate to select the final answer  
* **Multiple datasets**: StrategyQA, GSM‑8K, TruthfulQA, ScienceQA (the demo trains only on StrategyQA for speed)  

> **NOTE**  
> The full‑scale experiments of the original paper use many thousands of examples and fine‑tune on multiple tasks.  For the sake of speed and reproducibility in this lightweight version we train on a *small* subset (≈ 200 training examples) of the
> [StrategyQA](https://huggingface.co/datasets/strategyqa) dataset and evaluate on its test split (≈ 229 examples).

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── adapter.py
│   ├── trainer.py
│   ├── inference.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
├── reproduce.sh
└── README.md
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Create output directories.
3. Train a 0.1B‑parameter DeBERTa‑base adapter using ranking‑based NCE on StrategyQA.
4. Evaluate the adapter on the test split, printing the accuracy.

All outputs (model checkpoints, logs, and evaluation results) are written to the
`checkpoints/` and `logs/` directories.

## Expected Output

After training and evaluation you should see something like:

```
Epoch 1/5  loss=0.1234
  Test accuracy: 68.5%
Epoch 2/5  loss=0.1012
  Test accuracy: 70.2%
...
Training finished.
Inference accuracy: 68.7%
```

The exact number may vary slightly due to randomness but should be
close to the values reported in the paper for a small subset of the data.

## Extending to Other Tasks

To adapt the code to other datasets (GSM‑8K, TruthfulQA, ScienceQA) you
only need to change the `--dataset` argument in the training and
inference commands.  The core adapter and training logic remain the same.

## Baseline Comparison

The repository includes a simple zero‑shot baseline: the raw black‑box
LLM without any adapter.  For a fair comparison with the paper’s
SFT or LoRA baselines you can run the same pipeline without
loading an adapter.

## Reproducibility

All random seeds are fixed in the training script.  The code runs
on any standard Ubuntu 24.04 LTS Docker container with an NVIDIA
GPU (the GPU is optional; CPU will be used if no GPU is available).

---

### Common Issues

* **Tokenizers**: Some models require setting the `pad_token`.  The scripts
  handle this automatically.
* **GPU memory**: DeBERTa‑large (~304 M params) can be memory‑heavy.
  Reduce `--batch_size` or use `--adapter_model microsoft/deberta-v3-base`
  (86 M params) if you run out of memory.
* **LLM generation**: The inference script uses a pipeline that may
  download a large LLM (~8 B params).  Make sure you have enough disk space.

---

### Contact

For questions or feedback, please open an issue or contact the repository
maintainer.