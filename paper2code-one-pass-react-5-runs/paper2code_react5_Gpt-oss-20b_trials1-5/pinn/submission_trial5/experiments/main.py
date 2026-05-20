#!/usr/bin/env python3
"""
Full training pipeline for all PDEs, widths, optimizers, and seeds.

The script:
  * Samples training data.
  * Builds the MLP.
  * Trains with Adam, L‑BFGS, Adam+L‑BFGS, and NNCG.
  * Records loss and L2RE per iteration.
  * Computes top‑10 eigenvalues of the final Hessian.
  * Saves results to `results/<pde>_<width>_<opt>_<seed>.json`.
"""

import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple

from experiments import config
from experiments.models import MLP
from experiments import pinn
from experiments import utils
from experiments import hessian

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _train_adam(model: MLP,
                loss_fn,
                data: Dict[str, np.ndarray],
                lr: float,
                iterations: int,
                device: torch.device,
                log_interval: int = 1000) -> Tuple[List[float], List[float]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history, l2re_history = [], []

    for it in range(iterations):
        opt.zero_grad()
        loss = loss_fn(model, data)
        loss.backward()
        opt.step()

        if it % log_interval == 0 or it == iterations - 1:
            l2 = utils.evaluate_solution(model, data["pde"], data["eval_grid"])
            loss_history.append(loss.item())
            l2re_history.append(l2)

    return loss_history, l2re_history


def _train_lbfgs(model: MLP,
                 loss_fn,
                 data: Dict[str, np.ndarray],
                 iterations: int,
                 device: torch.device,
                 log_interval: int = 1000) -> Tuple[List[float], List[float]]:
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=config.LBFGS_LR,
        max_iter=20,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=config.LBFGS_MEM,
        line_search_fn=config.LBFGS_LINE_SEARCH,
    )
    loss_history, l2re_history = [], []

    def closure():
        opt.zero_grad()
        loss = loss_fn(model, data)
        loss.backward()
        return loss

    for it in range(iterations):
        opt.step(closure)

        if it % log_interval == 0 or it == iterations - 1:
            l2 = utils.evaluate_solution(model, data["pde"], data["eval_grid"])
            loss_history.append(closure().item())
            l2re_history.append(l2)

    return loss_history, l2re_history


def _train_adam_lbfgs(model: MLP,
                      loss_fn,
                      data: Dict[str, np.ndarray],
                      switch_iter: int,
                      lr_adam: float,
                      iterations: int,
                      device: torch.device,
                      log_interval: int = 1000) -> Tuple[List[float], List[float]]:
    opt_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=config.LBFGS_LR,
        max_iter=20,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=config.LBFGS_MEM,
        line_search_fn=config.LBFGS_LINE_SEARCH,
    )

    loss_history, l2re_history = [], []

    def adam_closure():
        opt_adam.zero_grad()
        loss = loss_fn(model, data)
        loss.backward()
        return loss

    def lbfgs_closure():
        opt_lbfgs.zero_grad()
        loss = loss_fn(model, data)
        loss.backward()
        return loss

    for it in range(iterations):
        if it < switch_iter:
            opt_adam.step(adam_closure)
        else:
            opt_lbfgs.step(lbfgs_closure)

        if it % log_interval == 0 or it == iterations - 1:
            l2 = utils.evaluate_solution(model, data["pde"], data["eval_grid"])
            if it < switch_iter:
                loss_history.append(adam_closure().item())
            else:
                loss_history.append(lbfgs_closure().item())
            l2re_history.append(l2)

    return loss_history, l2re_history


def _train_nncg(model: MLP,
                loss_fn,
                data: Dict[str, np.ndarray],
                iterations: int,
                device: torch.device,
                log_interval: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Very light‑weight NysNewton‑CG: a single damped Newton step per iteration
    using CG to solve (H + μI) d = -∇L.
    """
    loss_history, l2re_history = [], []

    mu = 1e-2  # damping
    for it in range(iterations):
        loss = loss_fn(model, data)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        # Define Hessian‑vector product
        def hv(v):
            v_list = torch.split(v, [p.numel() for p in model.parameters()])
            Hv = torch.autograd.grad(grad_flat, model.parameters(), grad_outputs=v_list, retain_graph=True)
            return torch.cat([h.reshape(-1) for h in Hv])

        # Solve (H + μI) d = -grad
        rhs = -grad_flat
        d, _ = torch.linalg.cg(hv, rhs, max_iter=20, atol=1e-6, rtol=1e-6, M=None)
        step = d / (torch.norm(d) + 1e-12)

        # Line search (Armijo)
        alpha = 1.0
        for _ in range(10):
            with torch.no_grad():
                for p, delta in zip(model.parameters(), step.split([p.numel() for p in model.parameters()])):
                    p.copy_(p + alpha * delta)
            new_loss = loss_fn(model, data)
            if new_loss <= loss + 1e-4 * alpha * (grad_flat @ step):
                break
            alpha *= 0.5

        loss_history.append(loss.item())
        l2 = utils.evaluate_solution(model, data["pde"], data["eval_grid"])
        l2re_history.append(l2)

    return loss_history, l2re_history


# ----------------------------------------------------------------------
# Loss function
# ----------------------------------------------------------------------
def pinn_loss(model: MLP, data: Dict[str, np.ndarray]) -> torch.Tensor:
    """
    Least‑squares PINN loss:
        0.5 / n_res * sum(residual^2) + 0.5 / n_bnd * sum(boundary^2)
    """
    pde = data["pde"]

    # Residual points
    res_pts = _to_tensor(data["res"], model.device)
    x_res = res_pts[:, 0]
    t_res = res_pts[:, 1]
    inp_res = torch.stack([x_res, t_res], dim=1)
    u_res = model(inp_res).squeeze()

    res = pde.residual(u_res, x_res, t_res)
    loss_res = 0.5 * (res ** 2).mean()

    # Initial condition
    init_pts = _to_tensor(data["init"], model.device)
    x_init = init_pts[:, 0]
    t_init = init_pts[:, 1]
    inp_init = torch.stack([x_init, t_init], dim=1)
    u_init = model(inp_init).squeeze()
    val_init = pde.analytical(x_init, t_init)
    loss_init = 0.5 * (pde.boundary(u_init, x_init, t_init, val_init) ** 2).mean()

    # Boundary conditions (both sides)
    bnd1_pts = _to_tensor(data["bnd1"], model.device)
    bnd2_pts = _to_tensor(data["bnd2"], model.device)

    def bnd_loss(pts, val):
        x_b = pts[:, 0]
        t_b = pts[:, 1]
        inp_b = torch.stack([x_b, t_b], dim=1)
        u_b = model(inp_b).squeeze()
        return 0.5 * (pde.boundary(u_b, x_b, t_b, val) ** 2).mean()

    loss_bnd1 = bnd_loss(bnd1_pts, val_init)  # value from analytical at that side
    loss_bnd2 = bnd_loss(bnd2_pts, val_init)

    loss_bnd = 0.5 * (loss_bnd1 + loss_bnd2)

    return loss_res + loss_init + loss_bnd


# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------
def run_experiment(pde_cls,
                   width: int,
                   optimizer: str,
                   seed: int,
                   switch_iter: int = None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample data
    data_raw = pde_cls.sample_points(
        config.RES_POINTS, config.INIT_POINTS, config.BND_POINTS
    )
    data_raw["pde"] = pde_cls
    data_raw["eval_grid"] = utils.create_evaluation_grid(pde_cls)

    # Build model
    model = MLP(input_dim=2, output_dim=1, width=width).to(config.DEVICE)
    model.device = config.DEVICE

    # Choose optimizer
    lr = config.get_adam_lr()
    if optimizer == "adam":
        loss_hist, l2_hist = _train_adam(
            model, pinn_loss, data_raw, lr, config.ITERATIONS, config.DEVICE
        )
    elif optimizer == "lbfgs":
        loss_hist, l2_hist = _train_lbfgs(
            model, pinn_loss, data_raw, config.ITERATIONS, config.DEVICE
        )
    elif optimizer == "adam+lbfgs":
        loss_hist, l2_hist = _train_adam_lbfgs(
            model, pinn_loss, data_raw, switch_iter, lr, config.ITERATIONS, config.DEVICE
        )
    elif optimizer == "nncg":
        loss_hist, l2_hist = _train_nncg(
            model, pinn_loss, data_raw, config.ITERATIONS, config.DEVICE
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Hessian spectral analysis (top‑10 eigenvalues)
    loss_final = torch.tensor(loss_hist[-1], device=config.DEVICE)
    params = list(model.parameters())
    eigs, _ = hessian.top_k_eigenvalues(loss_final, params, k=10, iters=10)
    cond = eigs[0] / max(eigs[-1], 1e-12)

    return {
        "loss": loss_hist,
        "l2re": l2_hist,
        "hessian_top10": eigs,
        "hessian_condition": cond,
        "switch_point": switch_iter,
    }


def main():
    os.makedirs("results", exist_ok=True)

    pdes = {
        "convection": pinn.Convection,
        "reaction": pinn.Reaction,
        "wave": pinn.Wave,
    }

    optimizers = ["adam", "lbfgs", "adam+lbfgs", "nncg"]

    summary = []

    for pde_name, pde_cls in pdes.items():
        for width in config.WIDTHS:
            for opt in optimizers:
                for seed in config.SEEDS:
                    switch = None
                    if opt == "adam+lbfgs":
                        for sp in config.SWITCH_POINTS:
                            res = run_experiment(pde_cls, width, opt, seed, sp)
                            fname = f"results/{pde_name}_{width}_{opt}_{seed}_{sp}.json"
                            with open(fname, "w") as f:
                                json.dump(res, f, indent=2)
                            summary.append(
                                f"{pde_name},{width},{opt},{seed},{sp},"
                                f"{res['loss'][-1]:.3e},{res['l2re'][-1]:.3e}"
                            )
                    else:
                        res = run_experiment(pde_cls, width, opt, seed)
                        fname = f"results/{pde_name}_{width}_{opt}_{seed}.json"
                        with open(fname, "w") as f:
                            json.dump(res, f, indent=2)
                        summary.append(
                            f"{pde_name},{width},{opt},{seed},-,"
                            f"{res['loss'][-1]:.3e},{res['l2re'][-1]:.3e}"
                        )

    # Save a human readable summary
    with open("results/summary.txt", "w") as f:
        f.write("pde,width,optimizer,seed,switch,final_loss,final_l2re\n")
        f.write("\n".join(summary))

    print("All experiments finished.  See results/ for details.")


if __name__ == "__main__":
    main()