# Stochastic Interpolants with Data‑Dependent Couplings  
Reproduction of the core experiments from  
> **Stochastic Interpolants with Data‑Dependent Couplings**  
> Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, Eric Vanden‑Eijnden  

## What is reproduced

* A lightweight implementation of the stochastic‑interpolant framework using a data‑dependent coupling (base is a noisy, partially observed version of the target).
* Two toy conditioning tasks:
  * **Image in‑painting** – random block masks on CIFAR‑10 images.
  * **Image super‑resolution** – down‑sampled (4×) images as conditioning.
* Training of a U‑Net velocity model by minimizing the quadratic regression loss described in the paper.
* Sampling by integrating the learned velocity field with a simple forward‑Euler solver.
* Evaluation of the synthetic samples against the real test set using Fréchet Inception Distance (FID).

> **NOTE**: The original paper uses ImageNet (256×256 / 512×512).  
> For a full‑scale reproduction you should replace the CIFAR‑10 data loader with an ImageNet data pipeline and increase the image resolution. The code structure below is fully compatible with that change.

## Folder structure

```
├── requirements.txt
├── README.md
├── reproduce.sh          # Full reproduction script
├── train.py              # Training loop
├── sample.py             # Sampling script
├── eval.py               # FID evaluation
├── utils.py              # Helper functions
├── models/
│   └── unet.py           # U‑Net velocity model
└── datasets/
    └── cifar10.py        # CIFAR‑10 data loader with mask / down‑sample utilities
```

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies.
2. Download CIFAR‑10.
3. Train the velocity model for 10 epochs.
4. Generate 5 000 samples for each task.
5. Compute and print the FID for both tasks.

All artefacts (checkpoint, generated images, FID scores) are stored in the `outputs/` directory.

## Expected outputs

* `outputs/checkpoints/velocity.pth` – trained U‑Net.
* `outputs/generated/inpainting/*.png` – in‑painted images.
* `outputs/generated/sr/*.png` – super‑resolved images.
* `outputs/eval.txt` – FID scores.

If the FID values are lower than the numbers reported in the paper, the implementation is working correctly.  
If the script fails, ensure that you are running on a machine with at least one NVIDIA GPU and that `torch` detects it (`torch.cuda.is_available()`).

## Extending to ImageNet

* Replace the `datasets/cifar10.py` import with your own `imagenet.py` that yields `(x1, mask)` pairs.
* Increase `img_size` in the U‑Net configuration (e.g., 256 or 512).
* Adjust the learning rate and batch size for the larger data (e.g., `batch_size=32`, `lr=1e-4`).
* Optionally use a multi‑GPU `torch.nn.DataParallel` wrapper.

Happy reproducing!