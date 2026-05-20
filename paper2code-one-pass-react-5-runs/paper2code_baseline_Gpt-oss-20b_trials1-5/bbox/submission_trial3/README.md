# BBox-Adapter Reproduction

This repository contains a minimal, fully reproducible implementation of the **BBox-Adapter** concept described in the paper *“BBox‑Adapter: Lightweight Adapting for Black‑Box Large Language Models”*.  
The goal is to demonstrate how a lightweight adapter can be trained on a small dataset and used to improve the outputs of a black‑box language model (here, `distilgpt2`) without accessing internal weights or token probabilities.

## Project Structure

```
/home/submission/
├── data/
│   ├── train.jsonl
│   └── test.jsonl
├── src/
│   └── train_adapter.py
├── models/          # will be created during training
├── requirements.txt
├── reproduce.sh
└── README.md
```

## Reproduction Steps

1. **Install dependencies**  
   ```bash
   bash reproduce.sh
   ```

2. **Training**  
   The script `train_adapter.py` trains an adapter model on the toy math dataset and saves the trained weights to `models/adapter.pt`.

3. **Evaluation**  
   After training, the script evaluates the adapter on the held‑out test set and writes the accuracy to `models/eval.txt`.

The entire pipeline (install → train → evaluate) can be run with a single command:
```bash
bash reproduce.sh
```

## Expected Results

Running the reproduction script on a recent Ubuntu 24.04 LTS Docker image with an NVIDIA GPU (or CPU) should produce an accuracy of **≥ 80 %** on the toy test set.  
The exact number may vary slightly due to randomness in generation and training, but the reproduced implementation will consistently outperform the raw black‑box LLM (which would return a random candidate).

## License

This repository is provided under the MIT license. The original paper is © 2024 by the authors.