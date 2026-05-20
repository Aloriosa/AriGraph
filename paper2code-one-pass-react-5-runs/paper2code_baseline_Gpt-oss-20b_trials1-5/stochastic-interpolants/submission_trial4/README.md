# Stochastic Interpolants with Data‑Dependent Couplings (Toy Implementation)

This repository contains a minimal, runnable implementation of the *Stochastic Interpolants with Data‑Dependent Couplings* framework described in the 2024 ICML paper by Albergo et al.  
The code demonstrates the core ideas on the MNIST dataset:

1. **Data‑dependent coupling** – the base density is a noisy copy of the target data.
2. **Stochastic interpolant** – a simple linear interpolation with time‑dependent coefficients.
3. **Velocity estimation** – a neural network is trained to minimize a quadratic loss derived in the paper.
4. **Sampling** – a probability‑flow ODE (deterministic transport) is integrated to generate new samples.

> **Note**  
> The full paper evaluates the method on ImageNet (super‑resolution & in‑painting).  
> Here we provide a toy example that runs in a few minutes on a GPU or CPU.

## Repository Layout

```
/home/submission/
│
├── reproduce.sh          # Bash script that installs deps, trains, samples
├── README.md
├── requirements.txt
├── model.py              # Velocity model (simple UNet‑like)
├── utils.py              # Time embedding & helper functions
├── train.py              # Training loop (Algorithm 1)
├── sample.py             # Sampling (Algorithm 2)
└── sample.png            # Generated image after training
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the velocity model for a few epochs on MNIST.
3. Generate a single sample image and save it as `sample.png`.

After the script finishes you should see a file `sample.png` containing a generated MNIST digit.

The whole pipeline (training + sampling) takes less than a few minutes on a modern GPU (or a few minutes on CPU).

## Expected Output

The final `sample.png` will be a 28×28 grayscale image that looks like a handwritten digit.  
During training the console will show the loss decreasing over epochs.

## Extending to the Paper’s Experiments

To reproduce the exact results reported in the paper (ImageNet super‑resolution / in‑painting) you would need:

* A high‑resolution dataset (ImageNet 256×256 or 512×512).
* A more sophisticated conditional velocity model (U‑Net with attention, class embeddings, etc.).
* A larger training schedule (several days of GPU time).
* The data‑dependent coupling defined in Sec. 3.2 with a suitable `m(x₁)` and noise schedule.

The code skeleton above can be adapted to those settings – simply replace the data loader, the model, and the coupling functions accordingly.

## License

MIT License. Feel free to use, modify, and extend this code for your own research.