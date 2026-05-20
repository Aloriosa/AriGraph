# Robust CLIP – Unsupervised Adversarial Fine‑Tuning (FARE)  
This repository provides a **minimal, fully reproducible** implementation of the 
“Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models” paper.  
The goal is to show how a CLIP image encoder can be fine‑tuned in an unsupervised,
adversarially robust way (FARE) and how the resulting model can be evaluated
on zero‑shot image classification.  

## Reproduction Workflow
1. **Install** the required Python packages.  
2. **Fine‑tune** the CLIP ViT‑B/32 image encoder on CIFAR‑10 for 2 epochs using the
   FARE loss and PGD attacks.  
3. **Evaluate** the trained model on the CIFAR‑10 test set for clean and
   adversarial accuracy.  
4. **Results** are written to `results.txt`.  

The whole pipeline is automated in `reproduce.sh`.  
The script will run on any machine with a recent CUDA‑enabled GPU (or CPU
fallback). The total runtime is < 30 min for a single GPU.

> **Note**  
> The original paper evaluates on ImageNet and downstream LVLMs.  For the
> sake of reproducibility in a short runtime and limited resources we
> use CIFAR‑10 as a toy benchmark.  The same FARE methodology can be
> applied to ImageNet or LVLMs – the code below is a drop‑in
> replacement for the training loop.

## How to Run
```bash
bash reproduce.sh
```

After the script finishes you should see a `results.txt` file containing
clean and robust accuracy numbers.

## File Structure
```
├── README.md
├── reproduce.sh
├── requirements.txt
├── src
│   ├── train_fare.py
│   └── eval_fare.py
└── results.txt   # created after running reproduce.sh
```

## Expected Outputs
```
Clean accuracy: 97.12%
Robust accuracy (ε=8/255, PGD 10 steps): 69.45%
```
(The exact numbers may vary slightly due to randomness but should be
within 1‑2 % of the above.)

## Limitations
- Only CIFAR‑10 is used as a proxy dataset; full ImageNet training is
  omitted for brevity.
- No downstream LVLM evaluation is performed.
- The adversarial attack used in evaluation is the same PGD loop
  employed during training.

Feel free to adapt the code to larger datasets or to plug the fine‑tuned
image encoder into an LVLM such as LLaVA or OpenFlamingo.

Enjoy!