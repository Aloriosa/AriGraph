# APT – Adaptive Pruning and Tuning (Minimal Reproduction)

This repository contains a lightweight, fully reproducible implementation of the APT method described in
> *APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference*.

The goal is to show the core ideas of APT in a short script that can run on a single NVIDIA GPU
within a few minutes.  The implementation is intentionally simplified compared to the full paper:
* only BERT‑base is used
* pruning is performed on attention heads
* LoRA‑style adapters (`APTAdapter`) are dynamically expanded
* a simple self‑distillation loop keeps a moving teacher copy

## How to run

```bash
./reproduce.sh
```

The script will:
1. Install dependencies (`torch`, `transformers`, `datasets`, `accelerate`)
2. Run `train_apt.py` which fine‑tunes BERT‑base on SST‑2
3. During training it prunes a few heads per epoch, expands the LoRA rank, and performs self‑distillation
4. At the end it evaluates on the test set and writes a `results.json` file

The final test accuracy is printed on the console and also stored in `results.json`.

## Repository layout

```
├── requirements.txt
├── reproduce.sh          # driver script
├── train_apt.py          # training loop + APT logic
├── README.md
└── results.json          # produced after training
```

## Notes

* The implementation is **not** a full re‑implementation of the paper – it is a minimal proof‑of‑concept
  that demonstrates the key algorithmic ideas and can be reproduced easily.
* Hyper‑parameters are chosen to keep the training time short (2 epochs on SST‑2).
* The pruning schedule is very simple: after every epoch the lowest‑salience 15 % of heads are masked out.
* LoRA rank grows by 8 per epoch up to a maximum of 32.
* Self‑distillation uses the previous epoch’s model as a teacher; the loss is a weighted sum of
  cross‑entropy and MSE between student and teacher logits.

Feel free to experiment with the parameters in `SST2Config` inside `train_apt.py` to explore different
pruning / tuning regimes.

---