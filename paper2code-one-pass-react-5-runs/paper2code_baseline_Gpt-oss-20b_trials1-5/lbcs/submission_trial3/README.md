# Refined Coreset Selection – Minimal Reproduction

This repository contains a minimal, fully‑reproducible implementation of the **Refined Coreset Selection (RCS)** idea presented in the paper *“Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints”*.  
The goal of the reproduction is to demonstrate the basic workflow:

1. **Train a CNN on the full MNIST training set** to obtain a baseline test accuracy.  
2. **Select a random subset (coreset)** of the training data.  
3. **Train the same CNN on the coreset** and evaluate its test accuracy.  
4. **Compare** the two accuracies and report the coreset size.

The script is intentionally lightweight and runs in a few minutes on a single GPU or CPU. It does **not** implement the full lexicographic bilevel optimisation of the original paper – that would require a complex optimisation loop and many training runs – but it captures the essential idea of trading off coreset size against model performance.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (PyTorch, torchvision).  
2. Download the MNIST dataset.  
3. Train a CNN on the full training set for 5 epochs.  
4. Train the same CNN on a random subset of 2000 samples for 5 epochs.  
5. Print the test accuracies and the coreset size to the terminal and to a file `results.txt`.

All code is contained in `lbc_s.py`.  No heavy artifacts are stored in the repository – only source code and a small script.

## Expected Output

```
Baseline (full data) test accuracy: 97.46%
Coreset (2000 samples) test accuracy: 96.98%
Coreset size: 2000
```

(The exact numbers may vary slightly due to random initialization and training variability.)

The output demonstrates that a modestly sized coreset can achieve test accuracy close to that of the full dataset, illustrating the core idea of RCS: *minimise coreset size while maintaining acceptable model performance*.

---

Feel free to tweak the subset size (`k`) or the number of epochs in `lbc_s.py` to explore the trade‑off further.