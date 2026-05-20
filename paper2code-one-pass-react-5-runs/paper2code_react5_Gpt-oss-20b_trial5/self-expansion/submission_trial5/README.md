# SEMA – Self‑Expansion of Pre‑trained Models with Mixture of Adapters

This repository contains a minimal but functional implementation of the **SEMA** continual learning method described in the paper *“Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning”*.  
The goal is to reproduce the core ideas:

* A frozen Vision Transformer (ViT‑B/16) backbone.  
* Modular adapters (two‑layer linear bottleneck) inserted after the MLP of selected transformer blocks.  
* Representation descriptors (AE) that monitor reconstruction error and trigger on‑demand expansion.  
* An expandable soft‑weighting router that mixes the outputs of all adapters in a layer.  
* A class‑incremental training loop on CIFAR‑100 (10 tasks × 10 classes).  

The code is intentionally lightweight to fit in the 1 GB limit and to run within a few hours on a single GPU.

## How to Run

```bash
# Make the reproduction script executable
chmod +x reproduce.sh

# Run the full pipeline
./reproduce.sh
```

The script will:

1. Install the required packages.  
2. Download CIFAR‑100.  
3. Train the SEMA model on 10 incremental tasks and print the accuracy after each task.

## Project Structure

```
.
├── config.yaml          # Hyper‑parameters
├── reproduce.sh         # Entry point
├── requirements.txt     # Dependencies
├── README.md            # This file
├── sema_train.py        # Main training script
├── utils.py             # Helper functions
├── data/cifar100_incr.py
├── models/
│   └── sema.py
```

## Key Components

### `models/sema.py`
- `SEMA` class: wraps a frozen ViT backbone, manages adapters, representation descriptors (RDs), and routers.
- `expand_if_needed`: decides whether to add a new adapter based on z‑scores of RD reconstruction errors.
- `forward`: passes input through the backbone, injects adapters via forward hooks, and produces logits.

### Training Procedure
1. **Scan Phase** – First pass over the new task to compute running statistics of RD losses.  
2. **Expansion Decision** – For each expandable layer, if all RDs have a z‑score above the threshold, a new adapter is added.  
3. **Training Phase** – Only the newly added adapters, RDs, and the classification head are trained for a few epochs.  
4. **Evaluation** – Accuracy on the test set of the current task.

### Representation Descriptor
Implemented as a small auto‑encoder (encoder + decoder). The reconstruction loss is used as the regularization term and for novelty detection.

### Router
A linear layer producing a softmax over the current number of adapters in a layer. The weighted sum of adapter outputs is added to the MLP output.

## Hyper‑parameters
All hyper‑parameters are defined in `config.yaml`. They follow the settings reported in the paper:

- `lr_adapter`: 0.005  
- `lr_rd`: 0.01  
- `num_epochs`: 5  
- `batch_size`: 32  
- `expansion_threshold`: 1.5 (z‑score)  
- `expand_layers`: last 3 transformer blocks (indices 7, 8, 9)  

Feel free to tweak these values.

## Limitations
- The implementation focuses on the core algorithmic idea.  
- It uses only CIFAR‑100 for demonstration; other datasets can be plugged in by extending `data.cifar100_incr`.  
- The code does not include logging, checkpointing, or detailed evaluation metrics beyond top‑1 accuracy.  

## Acknowledgements
The backbone model is obtained from `timm`.  
The paper’s authors: Huiyi Wang, Haodong Lu, Lina Yao, Dong Gong.  

Enjoy experimenting with SEMA!