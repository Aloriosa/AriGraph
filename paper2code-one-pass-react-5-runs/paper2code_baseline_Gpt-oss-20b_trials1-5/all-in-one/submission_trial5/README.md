# Simformer Reproduction

This repository contains a minimal, reproducible setup that clones the original
[Simformer](https://github.com/mackelab/simformer) repository, installs the
necessary dependencies, trains a Simformer on one of the benchmark tasks
(Gaussian linear) and runs inference on the trained model.  
All heavy artifacts (simulation data, trained checkpoints, etc.) are
generated on‑the‑fly, so the repository stays well below the 1 GB limit.

> **Important**  
> The reproduction script uses the official Simformer code from the
> `mackelab/simformer` GitHub repository.  No source code from that project
> is bundled in this repository to keep the size small; it is downloaded
> at runtime.

## How to use

```bash
# 1. Make the reproduction script executable
chmod +x reproduce.sh

# 2. Run the reproduction script
./reproduce.sh
```

The script will:
1. Install the required Python packages (PyTorch, JAX, hydra, sbi, etc.).
2. Clone the official Simformer repository.
3. Install Simformer as a local editable package.
4. Train a Simformer on the Gaussian linear benchmark task (10k simulations).
5. Run inference for a few synthetic observations.
6. Store the results in the `results/` directory.

After the script finishes, you can inspect the generated files in
`results/`.  The script prints a summary of the training loss and a few
sample posterior plots.

## Directory layout

```
/home/submission/
├─ reproduce.sh            # Main reproduction script
├─ README.md               # This file
└─ requirements.txt        # Optional list of packages
```

The `reproduce.sh` script is fully self‑contained and does not rely on any
hard‑coded absolute paths.  It can be executed from any location inside the
container.  It also works on a local machine with an NVIDIA GPU (CUDA
11.8+) or without a GPU (CPU fallback).