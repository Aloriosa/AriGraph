# Reproduction of “Stochastic Interpolants with Data‑Dependent Couplings”

This repository contains a minimal, fully‑automatic implementation of the core ideas from the paper
*Stochastic Interpolants with Data‑Dependent Couplings* (Albergo et al., 2024).

The implementation focuses on the following key points:

1. **Data‑dependent coupling**  
   The base sample is generated as  
   \[
   x_{0}=m(x_{1})+\sigma\,\varepsilon ,\qquad \varepsilon\sim\mathcal{N}(0,I)
   \]
   where `m(x1)` is task‑specific (low‑resolution image for super‑resolution,
   masked image for in‑painting).  This satisfies the coupling
   \(\rho(x_{0},x_{1})=\rho_{1}(x_{1})\,\rho_{0}(x_{0}\mid x_{1})\).

2. **Full stochastic interpolant**  
   \[
   I_{t}=(1-t)x_{0}+t\,x_{1}+\gamma_{t}z, \qquad
   \gamma_{t}=\sqrt{2t(1-t)},\;z\sim\mathcal{N}(0,I)
   \]
   with time derivative  
   \[
   \dot I_{t}=-x_{0}+x_{1}+\dot\gamma_{t}z,\qquad
   \dot\gamma_{t}=\frac{1-2t}{\sqrt{2t(1-t)}} .
   \]

3. **Velocity field learning** – the velocity network \(b_{t}(x,\xi)\) is trained by
   minimizing the quadratic loss  
   \[
   L=\mathbb{E}\!\left[\|\,b_{t}(I_{t},\xi)-\dot I_{t}\,\|^{2}\right].
   \]
   A small 3‑level UNet with sinusoidal time‑embedding and optional
   image‑shaped conditioning is used.

4. **Conditional modelling** – the network receives the current image,
   a sinusoidal time embedding, and a conditioning tensor
   (`mask` for in‑painting, `low‑res upsampled image` for super‑resolution).
   The baseline (independent coupling) has no conditioning.

5. **Probability‑flow ODE sampling** – after training the velocity field,
   the ODE  
   \[
   \dot X_{t}=b_{t}(X_{t},\xi)
   \]
   is solved from \(t=0\) to \(t=1\) using `torchdiffeq.odeint`.

6. **Evaluation** – 500 samples are generated and the Fréchet Inception Distance (FID)
   against the real CIFAR‑10 test set is reported.

The repository is intentionally lightweight: no pre‑downloaded heavy
artifacts are included, and all heavy computation is performed on the fly.
The training schedule, learning‑rate, and network architecture follow the
values reported in the paper as closely as possible while keeping the
runtime reasonable (< 7 days on a single A10 GPU).

## How to reproduce

```bash
# Install dependencies
bash reproduce.sh
```

The script will:

1. Train a baseline model (independent coupling) for the *in‑painting* task.
2. Train a coupled model (data‑dependent coupling) for the same task.
3. Generate 500 samples with each model.
4. Compute the FID for each set of samples and write them to
   `baseline_fid.txt` and `coupled_fid.txt`.

All checkpoints and generated images are saved under the directories
`baseline/`, `coupled/`, `samples_baseline/`, and `samples_coupled/`.

The same code can be used for the *super‑resolution* task by passing
`--task sr` to `train.py`.  For a full reproduction of the paper’s
ImageNet experiments a larger dataset and more training time are
required, but the procedure is identical.