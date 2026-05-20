"""
PINN implementation for the 1‑D wave equation

Equation:
    u_tt - 4 u_xx = 0,  x ∈ (0,1), t ∈ (0,1)
Boundary:
    u(0,t) = u(1,t) = 0
Initial:
    u(x,0) = sin(π x) + 0.5 sin(5π x)
    u_t(x,0) = 0

Analytical solution:
    u(x,t) = sin(π x) cos(2π t) + 0.5 sin(5π x) cos(10π t)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ------------------------------------------------------------
# Configurable hyperparameters
# ------------------------------------------------------------
WIDTH = 200            # hidden layer width
LEARNING_RATE = 1e-3   # Adam lr
ADAM_STEPS = 1000      # number of Adam steps before switching to LBFGS
TOTAL_STEPS = 20000    # total training steps (including LBFGS)
BATCH_SIZE = 1024      # used in Adam updates
SEED = 42

# ------------------------------------------------------------
# Set random seeds for reproducibility
# ------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Neural network definition
# ------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        # Xavier normal init for weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

# ------------------------------------------------------------
# Helper functions for derivatives
# ------------------------------------------------------------
def u_t(model, x, t):
    u = model(x, t).requires_grad_(True)
    return torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def u_tt(model, x, t):
    ut = u_t(model, x, t)
    return torch.autograd.grad(ut, t, grad_outputs=torch.ones_like(ut), create_graph=True)[0]

def u_x(model, x, t):
    u = model(x, t).requires_grad_(True)
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def u_xx(model, x, t):
    ux = u_x(model, x, t)
    return torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

# ------------------------------------------------------------
# Analytical solution for L2RE computation
# ------------------------------------------------------------
def analytical_solution(x, t):
    return torch.sin(np.pi * x) * torch.cos(2 * np.pi * t) + \
           0.5 * torch.sin(5 * np.pi * x) * torch.cos(10 * np.pi * t)

# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------
def generate_data():
    # Residual points: 10k random points in (0,1)x(0,1)
    n_res = 10000
    x_res = torch.rand(n_res, 1, device=device)
    t_res = torch.rand(n_res, 1, device=device)

    # Boundary points: x=0 and x=1, 101 points each in t
    t_b = torch.linspace(0., 1., 101, device=device).unsqueeze(1)
    x_b0 = torch.zeros_like(t_b)
    x_b1 = torch.ones_like(t_b)
    x_b = torch.cat([x_b0, x_b1], dim=0)
    t_b = torch.cat([t_b, t_b], dim=0)

    # Initial condition points: t=0, 257 points in x
    x_ic = torch.linspace(0., 1., 257, device=device).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)

    return (x_res, t_res), (x_b, t_b), (x_ic, t_ic)

# ------------------------------------------------------------
# Loss computation
# ------------------------------------------------------------
def pinn_loss(model, batch_res, batch_b, batch_ic):
    x_res, t_res = batch_res
    x_b, t_b = batch_b
    x_ic, t_ic = batch_ic

    # Residual loss
    u_tt_val = u_tt(model, x_res, t_res)
    u_xx_val = u_xx(model, x_res, t_res)
    res = u_tt_val - 4 * u_xx_val
    loss_res = torch.mean(res ** 2) / 2

    # Boundary loss
    u_b = model(x_b, t_b)
    loss_b = torch.mean(u_b ** 2) / 2

    # Initial condition loss
    u_ic_val = model(x_ic, t_ic)
    u_ic_t = u_t(model, x_ic, t_ic)
    u0 = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(5 * np.pi * x_ic)
    ut0 = torch.zeros_like(u0)
    loss_ic1 = torch.mean((u_ic_val - u0) ** 2) / 2
    loss_ic2 = torch.mean((u_ic_t - ut0) ** 2) / 2
    loss_ic = loss_ic1 + loss_ic2

    return loss_res + loss_b + loss_ic

# ------------------------------------------------------------
# L2 Relative Error on a fine grid
# ------------------------------------------------------------
def compute_l2re(model, n_x=255, n_t=100):
    x = torch.linspace(0., 1., n_x, device=device).unsqueeze(1)
    t = torch.linspace(0., 1., n_t, device=device).unsqueeze(1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    u_pred = model(X, T).detach()
    u_true = analytical_solution(X, T).detach()
    l2re = torch.norm(u_pred - u_true) / torch.norm(u_true)
    return l2re.item()

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train():
    model = PINN(WIDTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    (x_res, t_res), (x_b, t_b), (x_ic, t_ic) = generate_data()

    start_time = time.time()

    # Adam phase
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        loss = pinn_loss(model, (x_res, t_res), (x_b, t_b), (x_ic, t_ic))
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"Adam step {step+1}/{ADAM_STEPS} - loss: {loss.item():.3e}")

    # L-BFGS phase
    def closure():
        optimizer.zero_grad()
        loss = pinn_loss(model, (x_res, t_res), (x_b, t_b), (x_ic, t_ic))
        loss.backward()
        return loss

    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, max_eval=20,
                            history_size=10, line_search_fn='strong_wolfe')
    # We run LBFGS for several iterations
    for step in range(TOTAL_STEPS - ADAM_STEPS):
        loss = optimizer.step(closure)
        if (step + 1) % 200 == 0:
            print(f"L-BFGS step {step+1}/{TOTAL_STEPS-ADAM_STEPS} - loss: {loss.item():.3e}")

    training_time = time.time() - start_time
    final_loss = pinn_loss(model,
                           (x_res, t_res),
                           (x_b, t_b),
                           (x_ic, t_ic)).item()
    l2re = compute_l2re(model)

    print(f"\nTraining finished in {training_time:.1f} seconds")
    print(f"Final loss: {final_loss:.3e}")
    print(f"Final L2RE: {l2re:.3e}")

    # Save final predictions for inspection (optional)
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), "output/pinn_wave.pth")

if __name__ == "__main__":
    train()