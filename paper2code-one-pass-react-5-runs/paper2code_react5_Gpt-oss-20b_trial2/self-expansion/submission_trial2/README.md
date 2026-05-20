# SEMA – Self‑Expansion of Pre‑trained Models with Modularized Adapters

This repository contains a lightweight implementation of the **SEMA** continual learning framework described in:

> Huiyi Wang, Haodong Lu, Lina Yao, Dong Gong  
> *Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning*  
> 2024

The goal of this codebase is to reproduce the **class‑incremental learning** experiments on CIFAR‑100, ImageNet‑R, ImageNet‑A and VTAB as presented in the paper.  
The implementation focuses on the core ideas:

* frozen Vision Transformer (ViT) backbone  
* modular adapters (down/up projection) added per transformer layer on demand  
* representation descriptors (auto‑encoders) that monitor distribution shift  
* expandable weighting router that mixes the outputs of the adapters  
* self‑expansion logic driven by z‑score of reconstruction error  

> **NOTE** – This is a *minimal* implementation aimed at reproducing the paper’s high‑level behaviour.  
> It is **not** optimised for speed or memory usage and may not achieve the exact numbers reported in the paper.  
> For a production‑ready implementation, see the official project repository (when available).

## Directory structure

```
root/
├── requirements.txt
├── README.md
├── reproduce.sh          # entry point
├── src/
│   ├── dataset.py        # CIFAR‑100, ImageNet‑R, ImageNet‑A, VTAB loaders
│   ├── model.py          # ViT backbone + SEMA modules
│   ├── train.py          # training & evaluation loop
│   └── utils.py          # helpers
└── logs/                 # checkpoints & logs (created on run)
```

## Running

1. **Setup**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the reproduction script**  
   ```bash
   bash reproduce.sh
   ```

   The script will automatically download the required datasets, train the model on 10 incremental tasks (CIFAR‑100 example), evaluate after each task, and finally print a summary of accuracies.

3. **Logs & Checkpoints**  
   All checkpoints are stored under `logs/<dataset>/task_<n>/` and logs under `logs/<dataset>/train.log`.

## Customisation

* Change the dataset or number of tasks by editing `reproduce.sh`.  
* Hyper‑parameters are defined in `src/train.py` (`BATCH_SIZE`, `LR`, `EPOCHS`, etc.).  
* The expansion threshold (`EXPANSION_ZTHRESH`) can be tuned in `src/model.py`.

## Implementation notes

* **Adapters** – implemented as a two‑layer MLP with a down‑projection to a bottleneck (`DROPOUT_DIM`) and an up‑projection back to the original dimension.  
* **Representation Descriptors** – simple linear AE (`hidden_dim=64`).  
* **Router** – a linear layer producing a weight vector over the adapters in a given layer, followed by softmax.  
* **Self‑Expansion** – during the first epoch of a task we evaluate reconstruction errors of all frozen RDs.  
  If all RDs in a layer exceed the z‑score threshold, a new adapter/router is added to that layer.  
  Subsequent epochs train only the newly added modules, while older ones are frozen.  
* **Evaluation** – after each task we evaluate on the union of all seen classes.  
  The script records both *average accuracy* (`A_N`) and *average incremental accuracy* (`\overline{A}`).

Feel free to explore the code, tweak hyper‑parameters or extend to other datasets. Happy continual learning!