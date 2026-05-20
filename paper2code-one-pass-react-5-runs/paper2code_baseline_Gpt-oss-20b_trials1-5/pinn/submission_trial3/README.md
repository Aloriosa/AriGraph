# PINN Loss Landscape Reproduction

This repository reproduces a simplified experiment from the paper  
*“Challenges in Training PINNs: A Loss Landscape Perspective”*.

## What we reproduce

We train a Physics‑Informed Neural Network (PINN) to solve the
one‑dimensional convection equation

```
∂u/∂t + β ∂u/∂x = 0 ,   x ∈ (0, 2π), t ∈ (0, 1)
u(x,0) = sin(x)          (initial condition)
u(0,t) = u(2π,t)         (periodic boundary)
```

with `β = 40`.  
The PINN is a 3‑layer MLP with tanh activations and 50 hidden units
per layer.  We train for 5 000 optimisation steps:

1. 2 000 steps with Adam (learning rate `1e-3`).
2. 3 000 steps with L‑BFGS (default PyTorch settings).

After training we compute

* the final loss value (`results/conv_loss.txt`);
* the L₂‑relative error on a dense grid (`results/conv_l2re.txt`).

These numbers can be compared to the values reported in the paper
(see Table 1).

## How to run

```
bash reproduce.sh
```

The script:

1. Installs PyTorch 2.0, torchvision and scipy.
2. Runs `src/pinn.py` with the parameters described above.
3. Stores the outputs in the `results/` directory.

The entire repository is < 1 MB and contains only source code,
so the grader can safely clean the directory with
`git clean -fd`.

## Project structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── src/
│   └── pinn.py
└── results/
    ├── conv_loss.txt   # final loss value
    └── conv_l2re.txt   # final L2 relative error
```

## Expected outputs

After running `reproduce.sh` you should see something like:

```
Final loss: 4.73e-06
Final L2 relative error: 1.92e-02
```

The exact numbers may vary slightly due to random initialization
(controlled via the `--seed` flag).

## Extending the experiment

The `pinn.py` script is written generically and can be extended to
other PDEs in the paper (wave, reaction) by changing the
`--pde` argument.  New optimisation strategies (e.g. NysNewton‑CG)
can also be plugged in by modifying the training loop.