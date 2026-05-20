# SEMA – Self‑Expansion of Pre‑trained Models with Mixture of Adapters
This repository contains a minimal, runnable implementation of the **SEMA** method
(“Self‑Expansion of Pre‑trained Models with Mixture of Adapters”) for
class‑incremental learning on CIFAR‑10.

> **Note**  
> The implementation below is a lightweight demonstration that follows the
> high‑level algorithm described in the paper.  It is *not* a full‑fledged
> research implementation and therefore does not reproduce the exact
> performance numbers reported in the paper.  The goal is to provide a
> self‑contained, reproducible example that can be executed on a
> fresh Ubuntu 24.04 Docker container with an NVIDIA GPU.

## Repository structure
```
/home/submission/
├── README.md
├── reproduce.sh          # Entry script: installs deps, runs training
├── train_sema.py         # Minimal implementation of SEMA
└── output/               # Directory where results.json will be written
```

## How to run
```bash
bash reproduce.sh
```

The script will:
1. Install Python 3 and the required packages (`torch`, `torchvision`,
   `timm`, `tqdm`, `numpy`).
2. Download CIFAR‑10 and a pretrained ViT‑B/16 backbone.
3. Train the model on 5 incremental tasks (2 classes per task).
4. Evaluate after each task and write a `results.json` file in the
   `output/` folder.

After completion you can inspect the results:
```bash
cat output/results.json
```

## Expected outputs
The script prints progress for each task and epoch, e.g.:

```
===== TASK 1/5 =====
  → Adding new adapter and descriptor
  Task 1 Epoch 1: 100%|███████| 100/100 [00:12<00:00,  8.20it/s]
  ...
  Test accuracy on Task 1: 78.45%

===== TASK 2/5 =====
  Task 2 Epoch 1: 100%|███████| 100/100 [00:12<00:00,  8.23it/s]
  ...
  Test accuracy on Task 2: 70.12%
...
```

The final `results.json` contains per‑task accuracies, e.g.:
```json
{
  "per_task_accuracy": [
    78.45,
    70.12,
    65.78,
    60.34,
    55.21
  ]
}
```

## Code design
- **`SEMA` model**: holds a frozen ViT backbone, a list of adapters,
  representation descriptors (auto‑encoders) and a soft‑max router.
- **Self‑expansion**: during the first epoch of a new task, the
  reconstruction error of each descriptor is used to compute a z‑score.
  If *all* z‑scores exceed a threshold, a new adapter/descriptor pair is
  added to the last transformer block.
- **Training**: adapters and descriptors are trained together using a
  combined classification + reconstruction loss.  The backbone remains
  frozen throughout.
- **Evaluation**: accuracy is computed on the test set of each task
  after training.

## Extending the example
- To experiment with other datasets or more tasks, modify the
  `NUM_TASKS`, `CLASSES_PER_TASK`, and data loading logic in
  `train_sema.py`.
- To use a different pretrained backbone, replace the `timm.create_model`
  call with the desired model.
- The current implementation only expands adapters in the last
  transformer block for simplicity.  Expanding to multiple blocks would
  require extending the data structures accordingly.

## License
This repository is released under the MIT license.  The code is for
educational use only and is not a full implementation of the original
paper.