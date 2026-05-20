# Refined Coreset Selection (LBCS) – Reproduction

This repository contains a minimal, fully reproducible implementation of the
*Lexicographic Bilevel Coreset Selection* (LBCS) method described in the
paper *Refined Coreset Selection: Towards Minimal Coreset Size under Model
Performance Constraints*.

## Repository Layout

```
├── README.md
├── reproduce.sh            # Reproduction script
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point
│   ├── coreset_selection.py
│   ├── models.py
│   └── utils.py
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `torchvision`, `tqdm`, `numpy`).  
2. Download the MNIST dataset automatically.  
3. Run the LBCS algorithm on MNIST with a small convolutional network.  
4. Print the best coreset size found and the corresponding test accuracy.

The output should look similar to:

```
=== LBCS on MNIST ===
Predefined coreset size (k): 200
Performance compromise ε: 0.1
Total search iterations: 20

Best mask size: 185
Test accuracy: 97.12%
```

> **NOTE**: This is a *toy* implementation aimed at demonstrating the
> algorithmic idea and ensuring reproducibility. It does **not** reproduce
> the full experimental results reported in the paper, which require
> large‐scale training and a more sophisticated optimization pipeline.

## Repository Contents

- **src/main.py** – Orchestrates data loading, runs LBCS, and reports results.  
- **src/coreset_selection.py** – Implements the simplified lexicographic
  bilevel optimization (inner training + outer mask search).  
- **src/models.py** – Defines a small CNN suitable for MNIST.  
- **src/utils.py** – Helper functions (e.g., seed setting, mask handling).  

## License

MIT License – see `LICENSE` file for details.