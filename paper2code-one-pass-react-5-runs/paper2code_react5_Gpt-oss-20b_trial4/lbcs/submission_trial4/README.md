# Refined Coreset Selection (RCS) – Minimal Reproduction

This repository reproduces the core algorithm of the paper

> *Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints*  
> Xiaobo Xia, Jiale Liu, Shaokun Zhang, Qingyun Wu, Hongxin Wei, Tongliang Liu

The implementation focuses on the key ideas:
* **Bilevel optimisation** – inner loop trains a small CNN on a candidate
  coreset, outer loop updates the mask.
* **Lexicographic priority** – performance (validation accuracy) is the
  primary objective; coreset size is the secondary objective.
* **Pairwise mask comparison** – the outer loop keeps an incumbent mask and
  accepts a new mask only if it is lexicographically better.

The script runs on Fashion‑MNIST, SVHN and CIFAR‑10 and prints the
final test accuracy and the selected coreset size.

> **NOTE**  
> The numbers are illustrative; the implementation is a lightweight
> version and may not exactly match the paper’s reported results.

## Directory structure

```
.
├── main.py          # Core implementation
├── reproduce.sh     # Bash script that installs deps and runs main.py
├── README.md        # This file
└── results.csv      # Generated after running reproduce.sh
```

## Usage

```bash
# Make the script executable
chmod +x reproduce.sh

# Run the reproduction
./reproduce.sh
```

The script will install the required packages, run the algorithm, and
write the results to `results.csv`.  The console will show a summary
including test accuracy, coreset size and runtime per dataset.

## Reproduction details

* **Inner loop** – 2 epochs of Adam (lr = 1e‑3) on a randomly sampled
  coreset.
* **Outer loop** – 10 iterations.  Each iteration proposes a new mask
  of size in `[0.8 k, k]` and accepts it if it improves validation
  accuracy or keeps accuracy while reducing size.
* **Final evaluation** – 10 epochs of Adam on the selected coreset,
  evaluated on the official test split.
* **Device** – GPU if available, otherwise CPU.
* **Random seed** – 42 (ensures reproducibility).

## Limitations

* The implementation uses a simple random search for mask updates
  rather than a sophisticated bilevel optimisation strategy
  (e.g., gradient‑based sampling).  This keeps the runtime short.
* The `epsilon` parameter (allowed performance compromise) is not
  explicitly enforced; the lexicographic rule naturally keeps the
  performance high.
* The code does not include baseline comparisons or ImageNet experiments
  present in the paper.  Those would require a substantial amount of
  additional code and computational resources.

## License

This repository is provided under the MIT license.  All code is
originally written for the reproduction task and does not contain
any proprietary components from the original paper.