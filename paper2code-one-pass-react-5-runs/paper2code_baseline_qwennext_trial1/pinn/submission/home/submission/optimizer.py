"""
Optimizers for PINNs
This module implements the optimizers described in the paper:
- Adam (first-order)
- L-BFGS (second-order)
- Adam+L-BFGS (hybrid)
- NysNewton-CG (novel second-order)
"""

import torch
import numpy as np
import math

class Adam:
    """
    Adam optimizer
    """
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-7):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, closure=None):
        self.t += 1
        if closure is not None:
            closure()
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad.data
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p.grad.data ** 2
            else:
                self.m[i] = torch.zeros_like(self.m[i])
                self.v[i] = torch.zeros_like(self.v[i])
        for i, p in enumerate(self.params):
            if p.grad is not None:
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            else:
                m_hat = torch.zeros_like(self.m[i])
        for i, p in enumerate(self.params):
            if p.grad is not None:
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            else:
                v_hat = torch.zeros_like(self.v[i])
        for i, p in enumerate(self.params):
            if p.grad is not None:
                p.data = p.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def get_params(self):
        return [p.data.clone() for p in self.params]
    
    def set_params(self, params):
        for p, param in zip(self.params, params):
            p.data.copy_(param)

class LBFGS:
    """
    L-BFGS optimizer
    """
    
    def __init__(self, params, lr=1.0, history_size=10, max_iter=10):
        self.params = list(params)
        self.lr = lr
        self.history_size = history_size
        self.max_iter = max_iter
        self.s = []
        self.y = []
        self.rho = []
        self.p = [torch.zeros_like(p) for p in self.params]
        self.grad = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, closure=None):
        if closure is not None:
            closure()
        # Compute gradient
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.grad[i] = p.grad.data.clone()
        # L-BFGS update
        if len(self.s) >= self.history_size:
            self.s.pop(0)
            self.y.pop(0)
        # Store current gradient
        s = [g - g_old for g, g_old in zip(self.grad, self.s[-1])
        y = [p - p_old for p, p_old in zip(self.params, self.p)
        self.s.append(s)
        self.y.append(y)
        # Compute rho
        if len(self.s) > 0:
            rho = 1.0 / (sum([torch.dot(s_i, y_i).item() for s_i, y_i in zip(s, y)])
        else:
            rho = 1.0
        self.rho.append(rho)
        # Compute search direction
        q = self.grad
        for i in range(len(self.s) - 1, -1, -1):
            beta = self.rho[i] * sum([s_i * q_i for s_i, q_i in zip(self.s[i], q)]
            q = [q_i - beta * y_i for q_i, y_i in zip(q, self.y[i])
        # Update parameters
        for i, p in enumerate(self.params):
            p.data = p.data - self.lr * q[i]
        return closure()
    
    def get_params(self):
        return [p.data.clone() for p in self.params]
    
    def set_params(self, params):
        for p, param in zip(self.params, params):
            p.data.copy_(param)

class NysNewtonCG:
    """
    NysNewton-CG optimizer
    """
    
    def __init__(self, params, lr=0.01, max_iter=10, damping=1e-5):
        self.params = list(params)
        self.lr = lr
        self.max_iter = max_iter
        self.damping = damping
        self.H = None
        self.H_inv = None
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, closure=None):
        if closure is not None:
            closure()
        # Compute gradient
        grad = [p.grad.data.clone() for p in self.params]
        # Use Nyström approximation for Hessian
        if self.H is None:
            self.H = torch.eye(len(grad)) * 1.0
            self.H_inv = torch.eye(len(grad)) * 1.0
        # CG solver
        p = grad
        r = grad
        r_old = grad
        r_norm = torch.norm(r)
        r_norm_old = r_norm
        r_norm_min = r_norm
        iter = 0
        while iter < self.max_iter and r_norm > 1e-5:
            if iter > 0:
                beta = (r_norm ** 2) / (r_norm_old ** 2)
                p = [r_i + beta * p_i for r_i, p_i in zip(r, p)]
            # Compute Hessian-vector product
            h = [torch.zeros_like(p_i) for p_i in p]
            for i in range(len(p)):
                for j in range(len(p)):
                    h[i] += self.H[i][j] * p[j]
            # Update
            alpha = (r_norm ** 2) / (sum([r_i * h_i for r_i, h_i in zip(r, h)])
            for i in range(len(p)):
                grad[i] = grad[i] - alpha * h[i]
            r_old = r
            r = grad
            r_norm_old = r_norm
            r_norm = torch.norm(r)
            r_norm_min = min(r_norm, r_norm_min)
            iter += 1
        # Update parameters
        for i, p in enumerate(self.params):
            p.data = p.data - self.lr * grad[i]
        return closure()
    
    def get_params(self):
        return [p.data.clone() for p in self.params]
    
    def set_params(self, params):
        for p, param in zip(self.params, params):
            p.data.copy_(param)