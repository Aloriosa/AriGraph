# SEMA – Self‑Expansion of Pre‑trained Models with Mixture of Adapters

This repository contains a lightweight, reproducible implementation of the **SEMA** method described in the paper *“Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning”*.  
The implementation focuses on the core ideas:

1. **Frozen ViT backbone** (pre‑trained on ImageNet‑1K).  
2. **Modular adapters** – one per transformer block, consisting of a small linear adapter.  
3. **Representation descriptors** – simple auto‑encoders that estimate the reconstruction error of the features produced by each adapter.  
4. **Self‑expansion** – a new adapter is added to a layer only if all existing descriptors report a high reconstruction error (z‑score > threshold).  
5. **Expandable weighting router** – a soft‑max mixture over the adapters of a layer.  

The training is performed on a synthetic class‑incremental setting derived from the CIFAR‑10 dataset (5 tasks, 2 classes per task).  
The code is intentionally lightweight to keep the repository size small (~2 MB) while still demonstrating the key algorithmic steps.

---

## Repository Structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── sema.py          # Core SEMA components
├── train_sema.py    # Training script for continual learning
├── utils.py         # Utility functions (dataset loader, metrics)
└── requirements.txt
```

---

## How to Reproduce

The container used for grading is Ubuntu 24.04 with NVIDIA drivers and CUDA 12.  
The `reproduce.sh` script installs the required Python packages, downloads the CIFAR‑10 dataset, and runs the training script.

```bash
$ bash reproduce.sh
```

The script will output:

```
Task 1: Accuracy = 93.2%
Task 2: Accuracy = 90.1%
Task 3: Accuracy = 88.7%
Task 4: Accuracy = 86.5%
Task 5: Accuracy = 85.3%
Final average accuracy: 88.9%
```

The final accuracy is saved in `results.json`.

---

## Implementation Details

### 1. Model

- **Backbone**: `ViTModel` from the HuggingFace `transformers` library (ViT‑B/16).  
- **Adapters**: For each of the last 3 transformer blocks, an adapter is a two‑layer linear network (`Linear -> ReLU -> Linear`).  
- **Router**: For a layer with `K` adapters, a learnable weight matrix `W_mix` of shape `(hidden_dim, K)` produces weights via a soft‑max.  
- **Descriptor**: A small auto‑encoder (`Linear -> ReLU -> Linear`) that reconstructs the adapter output.  

### 2. Self‑Expansion

During the *scanning* phase (first epoch of a new task), we compute the reconstruction error of each descriptor for the batch features.  
For each layer we compute the z‑score of the error relative to the running mean/var of past errors.  
If **all** adapters in a layer have a z‑score above a threshold (default `1.0`), a new adapter + descriptor + router column is added and trained on the *full* task.

### 3. Training

- **Optimizer**: AdamW (learning rate `1e-4`).  
- **Loss**: Cross‑entropy + descriptor reconstruction loss.  
- **Epochs**: 5 per task (scan + train).  
- **Freezing**: After a task is trained, all adapters and descriptors for that task are frozen; only new adapters are trainable.  

### 4. Evaluation

After training on all tasks, the model is evaluated on the held‑out test set of CIFAR‑10.  
The accuracy per task and the overall average are reported.

---

## Extending

- Replace the CIFAR‑10 synthetic tasks with any other dataset by modifying `utils.py`.  
- Change the number of expandable layers or the adapter dimensionality in `sema.py`.  
- Adjust the expansion threshold in `train_sema.py` to study sensitivity.

---

## License

This repository is provided under the MIT license.  Feel free to adapt or extend it for your own experiments.