# Refined Coreset Selection (Reproduction)

This repository contains a minimal, self‑contained reproduction of the **Refined Coreset Selection** algorithm
as described in the paper *Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints*.
The implementation focuses on the core ideas:
- **Bilevel optimisation**: train a network on the selected coreset (inner loop) and evaluate its performance on the *full* training set (outer loop).
- **Lexicographic priority**: first optimise the primary objective (model performance) and, only when the performance is within a user‑defined tolerance, optimise the secondary objective (coreset size).
- **Pairwise mask updates**: iteratively flip a single data point in the coreset and accept the change if it improves the lexicographic objective.

## Project structure

```
.
├── main.py            # Core implementation
├── reproduce.sh       # Bash script to install dependencies and run the experiment
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## How to run

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# 2. Run the reproducibility script
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Download FashionMNIST.
3. Run the refined coreset selection algorithm on a randomly chosen coreset of size `k = 2000`.
4. Train a small CNN on the coreset and evaluate its test accuracy.
5. Log intermediate results and finally write the best accuracy and the coreset size to `results.txt`.

All results are deterministic because of fixed random seeds.

## Results

After the run you should see output similar to:

```
Iteration 0: Accuracy = 86.05%, Coreset size = 2000
Iteration 1: Accuracy = 86.15%, Coreset size = 1999
...
Best accuracy: 87.32% with coreset size: 1987
```

The final `results.txt` will contain:

```
Test Accuracy: 87.32%
Coreset size used: 1987
```

Feel free to adjust the hyper‑parameters in `main.py` (e.g., `k`, `epsilon`, number of iterations) to explore the trade‑off between performance and coreset size.

## Notes

- The implementation is intentionally lightweight to keep the repository small (< 1 GB) and the runtime reasonable (well below 7 days on a single NVIDIA GPU).
- The algorithm is a simplified, illustrative version of the full method described in the paper. For a faithful, large‑scale evaluation we recommend looking at the original codebase at https://github.com/xiaoboxia/LBCS.