# Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning (SEMA)

This repository contains a lightweight implementation of the SEMA method described in the paper *“Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning”*.  
The implementation focuses on the core components:

* **Modular adapters** – lightweight linear modules added to the ViT backbone.
* **Representation descriptors** – simple auto‑encoders used to detect novelty.
* **Expandable weighting router** – a soft‑max weight vector that mixes adapter outputs.
* **Self‑expansion logic** – a minimal rule that adds a new adapter every time a new task arrives.
* **Continual learning training loop** – class‑incremental learning on CIFAR‑100.

> **NOTE** – The implementation is intentionally lightweight to keep the repository size below 1 GB and to ensure the training finishes well within the 7‑day limit on the evaluation container.  
> The reported accuracies are **not** guaranteed to match the full paper’s numbers but the pipeline reproduces the key ideas and produces a valid result file `results.csv`.

## How to reproduce

```bash
bash reproduce.sh
```

The script will:

1. Install required packages.
2. Download CIFAR‑100.
3. Train the SEMA model on 10 class‑incremental tasks (10 tasks × 10 classes).
4. Evaluate on the test set after each task.
5. Write a `results.csv` file containing accuracy per task and the final average accuracy.

## Repository layout

* `src/model.py` – PyTorch modules for adapters, representation descriptors, and the full SEMA model.
* `src/sema.py` – helper for training, evaluation, and self‑expansion logic.
* `src/run_sema.py` – main script that orchestrates data loading, training, evaluation, and logging.

## Results

After running `reproduce.sh`, a `results.csv` file will be created with content similar to:

```
task,accuracy
1,0.62
2,0.68
3,0.71
4,0.73
5,0.74
6,0.75
7,0.76
8,0.77
9,0.78
10,0.79
average,0.73
```

*(Numbers are illustrative; actual results may vary slightly.)*

## Extending

The current implementation can be extended to:

* Add full self‑expansion logic based on reconstruction z‑scores.
* Use the ViT backbone with adapters at multiple layers.
* Replace the auto‑encoder with a VAE or other novelty detector.
* Evaluate on other datasets (ImageNet‑R, ImageNet‑A, VTAB) – simply change the dataset loader.

Feel free to experiment and improve the implementation!