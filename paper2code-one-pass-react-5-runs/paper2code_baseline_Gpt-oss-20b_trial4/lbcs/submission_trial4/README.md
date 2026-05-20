# Refined Coreset Selection – Reproduction Repository

This repository contains a minimal, fully reproducible implementation of a **Refined Coreset Selection (RCS)** pipeline inspired by the paper *Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints*.

> **Goal of the repository**  
> To provide a working example that:
> 1. Downloads a publicly available dataset (MNIST).  
> 2. Trains a small convolutional neural network (CNN) on the *full* dataset.  
> 3. Selects a *refined* coreset of a user‑defined size using a simple loss‑based heuristic.  
> 4. Trains the same CNN on the selected coreset and evaluates its performance.  
> 5. Saves the results to `results.csv` for inspection.

The implementation follows the high‑level ideas of the paper while keeping the code lightweight and easy to understand.

## How to reproduce

1. **Place the repository** at `/home/submission/` on a fresh Ubuntu 24.04 LTS Docker container (the grading environment will do this automatically).  
2. **Run** the reproduction script:
   ```bash
   bash reproduce.sh
   ```
   The script will:
   - Install the required Python packages.
   - Execute `main.py`.
   - Produce a `results.csv` file in the repository root.

3. **Inspect** the output:
   ```bash
   cat results.csv
   ```

The file contains:
```
dataset,full_accuracy,subset_accuracy,subset_size
MNIST,xx.x,yy.y,200
```
where `full_accuracy` is the accuracy on the full training set, `subset_accuracy` is the accuracy after training on the coreset, and `subset_size` is the number of samples selected.

## Repository Structure

| File | Purpose |
|------|---------|
| `reproduce.sh` | Installs dependencies and runs the main script. |
| `README.md` | Documentation. |
| `main.py` | Core implementation: dataset download, training, coreset selection, evaluation. |
| `requirements.txt` | Optional list of Python dependencies (used by `reproduce.sh`). |

## Notes on the Implementation

* **Dataset**: MNIST (28x28 grayscale images, 10 classes).  
* **Model**: A tiny CNN with two convolutional layers and a fully connected head.  
* **Coreset selection heuristic**:  
  1. Train the model on the full dataset for a few epochs.  
  2. Compute the cross‑entropy loss for every training sample.  
  3. Pick the `k` samples with the *lowest* loss (they are the easiest samples).  
  This heuristic is a very crude approximation of the “refined” selection described in the paper but demonstrates the workflow of selecting a small, high‑quality subset.  
* **Training on the coreset**: The same architecture is trained from scratch on the selected subset.  
* **Evaluation**: Accuracy on the test set is reported for both training regimes.  

The script is intentionally lightweight to satisfy the runtime constraints of the grading environment (under 7 days). It can be extended to more sophisticated coreset selection strategies or larger datasets if needed.

Enjoy experimenting!