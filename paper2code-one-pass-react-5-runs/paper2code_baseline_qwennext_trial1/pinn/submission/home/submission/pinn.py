#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) implementation for solving PDEs
This code reproduces the experiments from the paper:
"Challenges in Training PINNs: A Loss Landscape Perspective"
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from tqdm import tqdm
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs
    This is a multi-layer perceptron with tanh activations
    The network takes spatial coordinates as input and outputs the solution
    """
    
    def __init__(self, input_dim=2, hidden_layers=[50, 50, 50], output_dim=1, activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
        #     # Xavier initialization for weights
        #     nn.init.xavier_uniform_(layers[-1].weight)
        #     nn.init.zeros_(layers[-1].bias)
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        # nn.init.xavier_uniform_(layers[-1].weight)
        # nn.init.zeros_(layers[-1].bias)
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        """
        Forward pass
        Input: x of shape (batch_size, input_dim)
        Output: prediction of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def predict(self, x):
        """
        Make prediction
        """
        self.eval()
        with torch.no_grad():
            return self(x)
    
    def get_weights(self):
        """
        Get network weights
        """
        return [param.data.clone() for param in self.parameters()]
    
    def set_weights(self, weights):
        """
        Set network weights
        """
        for param, weight in zip(self.parameters(), weights):
            param.data.copy_(weight)
    
    def get_grad_norm(self):
        """
        Get gradient norm
        """
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

class PDESolver:
    """
    Solver for PDEs using PINNs
    """
    
    def __init__(self, pde_type='wave', pde_params=None):
        self.pde_type = pde_type
        self.pde_params = pde_params or {}
        self.setup_pde()
        
    def setup_pde(self):
        """Setup PDE based on type"""
        if self.pde_type == 'convection':
            # Convection PDE: u_t + beta * u_x = 0
            self.beta = self.pde_params.get('beta', 40.0)
            self.domain_x = (0, 2 * np.pi)
            self.domain_t = (0, 1.0)
            self.bc_type = 'periodic'
            self.initial_condition = lambda x: np.sin(x)
            self.analytical_solution = lambda x, t: np.sin(x - self.beta * t)
            
        elif self.pde_type == 'reaction':
            # Reaction ODE: u_t - rho * u * (1 - u) = 0
            self.rho = self.pde_params.get('rho', 5.0)
            self.domain_x = (0, 2 * np.pi)
            self.domain_t = (0, 1.0)
            self.bc_type = 'periodic'
            self.initial_condition = lambda x: np.exp(-(x - np.pi)**2 / (2 * (np.pi/4)**2))
            self.analytical_solution = lambda x, t: (self.initial_condition(x) * np.exp(self.rho * t)) / (self.initial_condition(x) * np.exp(self.rho * t) + 1 - self.initial_condition(x))
            
        elif self.pde_type == 'wave':
            # Wave PDE: u_tt - 4 * u_xx = 0
            self.beta = self.pde_params.get('beta', 5.0)
            self.domain_x = (0, 1.0)
            self.domain_t = (0, 1.0)
            self.bc_type = 'dirichlet'
            self.initial_condition = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(self.beta * np.pi * x)
            self.analytical_solution = lambda x, t: np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(self.beta * np.pi * x) * np.cos(2 * self.beta * np.pi * t)
        
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")
        
        # Setup discretization
        self.n_res = 10000
        self.n_bc = 257
        self.n_init = 257
        self.n_bound = 101
        self.setup_discretization()
        
    def setup_discretization(self):
        """Setup discretization grid"""
        # Residual points
        x_res = np.linspace(self.domain_x[0], self.domain_x[1], int(np.sqrt(self.n_res)))
        t_res = np.linspace(self.domain_t[0], self.domain_t[1], int(np.sqrt(self.n_res)))
        X_res, T_res = np.meshgrid(x_res, t_res)
        self.res_points = np.column_stack([X_res.flatten(), T_res.flatten()])
        
        # Boundary/initial points
        if self.pde_type == 'convection' or self.pde_type == 'reaction':
            # Periodic boundary
            x_bc = np.linspace(self.domain_x[0], self.domain_x[1], self.n_bc)
            t_bc = np.linspace(self.domain_t[0], self.domain_t[1], self.n_bc)
            X_bc, T_bc = np.meshgrid(x_bc, t_bc)
            self.bc_points = np.column_stack([X_bc.flatten(), T_bc.flatten()])
            
            # Initial points
            x_init = np.linspace(self.domain_x[0], self.domain_x[1], self.n_init)
            t_init = np.ones(self.n_init) * self.domain_t[0]
            self.init_points = np.column_stack([x_init, t_init])
            
        elif self.pde_type == 'wave':
            # Dirichlet boundary
            x_bc = np.linspace(self.domain_x[0], self.domain_x[1], self.n_bc)
            t_bc = np.linspace(self.domain_t[0], self.domain_t[1], self.n_bc)
            X_bc, T_bc = np.meshgrid(x_bc, t_bc)
            self.bc_points = np.column_stack([X_bc.flatten(), T_bc.flatten()])
            
            # Initial points
            x_init = np.linspace(self.domain_x[0], self.domain_t[1], self.n_init)
            t_init = np.ones(self.n_init) * self.domain_t[0]
            self.init_points = np.column_stack([x_init, t_init])
        
        # Convert to tensors
        self.res_points = torch.tensor(self.res_points, dtype=torch.float32).to(device)
        self.bc_points = torch.tensor(self.bc_points, dtype=torch.float32).to(device)
        self.init_points = torch.tensor(self.init_points, dtype=torch.float32).to(device)
        
        # Compute derivatives for analytical solution
        self.res_points.requires_grad = True
        self.bc_points.requires_grad = True
        self.init_points.requires_grad = True
        
    def compute_residual(self, model, x):
        """
        Compute PDE residual
        """
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        if self.pde_type == 'convection':
            # u_t + beta * u_x = 0
        # Compute gradients
            u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
            residual = u_t + self.beta * u_x
            
        elif self.pde_type == 'reaction':
            # u_t - rho * u * (1 - u) = 0
            u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
            residual = u_t - self.rho * u * (1 - u)
            
        elif self.pde_type == 'wave':
            # u_tt - 4 * u_xx = 0
            u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
            u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
            residual = u_tt - 4 * u_xx
        
        return residual
    
    def compute_boundary_condition(self, model, x):
        """
        Compute boundary condition residual
        """
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        if self.pde_type == 'convection' or self.pde_type == 'reaction':
            # Periodic boundary: u(0,t) = u(2*pi,t)
            u_bc = model(x)
            u_bc_0 = u_bc[:, 0]
            u_bc_1 = u_bc[:, 1]
            residual = u_bc_0 - u_bc_1
        elif self.pde_type == 'wave':
            # Dirichlet boundary: u(0,t) = 0
            u_bc = model(x)
            u_bc = u_bc[:, 0]
            residual = u_bc
        
        return residual
    
    def compute_initial_condition(self, model, x):
        """
        Compute initial condition residual
        """
        x = x.clone().requires_grad_(True)
        u = model(x)
        
        if self.pde_type == 'convection' or self.pde_type == 'reaction':
            # Initial condition: u(x,0) = sin(x)
            u_init = model(x)
            u_init = u_init[:, 0]
        elif self.pde_type == 'wave':
            # Initial condition: u(x,0) = sin(pi*x) + 0.5*sin(beta*pi*x)
            u_init = model(x)
            u_init = u_init[:, 0]
        
        return u_init
    
    def compute_loss(self, model, res_points, bc_points, init_points):
        """
        Compute PINN loss
        """
        # PDE residual loss
        residual = self.compute_residual(model, res_points)
        loss_res = torch.mean(residual ** 2)
        
        # Boundary condition loss
        bc_res = self.compute_boundary_condition(model, bc_points)
        loss_bc = torch.mean(bc_res ** 2)
        
        # Initial condition loss
        init_res = self.compute_initial_condition(model, init_points)
        loss_init = torch.mean(init_res ** 2)
        
        # Total loss
        loss = loss_res + loss_bc + loss_init
        
        return loss, loss_res, loss_bc, loss_init

class AdamOptimizer:
    """
    Adam optimizer for PINNs
    """
    
    def __init__(self, pinn, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.pinn = pinn
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(param.data) for param in self.pinn.parameters()]
        self.v = [torch.zeros_like(param.data) for param in self.pinn.parameters()]
        self.t = 0
    
    def step(self, res_points, bc_points, init_points, solver):
        """
        Perform one optimization step
        """
        self.t += 1
        self.pinn.train()
        self.pinn.zero_grad()
        
        loss, loss_res, loss_bc, loss_init = solver.compute_loss(self.pinn, res_points, bc_points, init_points)
        
        loss.backward()
        
        # Update Adam parameters
        for i, param in enumerate(self.pinn.parameters()):
            if param.grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad.data ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            else:
                self.m[i] = torch.zeros_like(self.m[i])
                self.v[i] = torch.zeros_like(self.v[i])
                m_hat = torch.zeros_like(self.m[i])
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data = param.data - self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        return loss.item()

class LBFGSOptimizer:
    """
    L-BFGS optimizer for PINNs
    """
    
    def __init__(self, pinn, learning_rate=1.0, history_size=10, max_iter=10):
        self.pinn = pinn
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.max_iter = max_iter
        self.s = []
        self.y = []
        self.rho = []
        self.p = [torch.zeros_like(param.data) for param in self.pinn.parameters()]
    
    def step(self, res_points, bc_points, init_points, solver):
        """
        Perform one optimization step
        """
        self.pinn.train()
        self.pinn.zero_grad()
        
        def closure():
            loss, loss_res, loss_bc, loss_init = solver.compute_loss(self.pinn, res_points, bc_points, init_points)
            loss.backward()
            return loss.item()
        
        loss = closure()
        
        # L-BFGS update
        if len(self.s) >= self.history_size:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        
        # Store current gradient
        grad = [param.grad.data.clone() for param in self.pinn.parameters()]
        
        if len(self.s) > 0:
            s = [g - g_old for g, g_old in zip(grad, self.s[-1])]
            y = [param.data - param_old for param, param_old in zip(self.p, self.p)]
            self.s.append(s)
            self.y.append(y)
            self.rho.append(1.0 / (sum([torch.dot(s_i, y_i).item() for s_i, y_i in zip(s, y)])))
        
        # Compute search direction
        q = [grad_i.clone() for grad_i in grad]
        alpha = 1.0
        for i in range(len(self.s) - 1, -1, -1):
            if len(self.rho) > i:
                beta = self.rho[i] * sum([s_i * q_i for s_i, q_i in zip(self.s[i], q)])
            else:
                beta = 0.0
            q = [q_i - beta * y_i for q_i, y_i in zip(q, self.y[i])
        # Compute Hessian
        if len(self.s) > 0:
            h = [self.learning_rate * q_i for q_i in q]
        else:
            h = [self.learning_rate * grad_i for grad_i in grad]
        
        # Update parameters
        for param, h_i in zip(self.p, h):
            param.data = param.data - h_i
        
        return loss

class NysNewtonCG:
    """
    NysNewton-CG optimizer for PINNs
    """
    
    def __init__(self, pinn, learning_rate=0.01, max_iter=10, damping=1e-5):
        self.pinn = pinn
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.damping = damping
        self.H = None
        self.H_inv = None
    
    def step(self, res_points, bc_points, init_points, solver):
        """
        Perform one optimization step
        """
        self.pinn.train()
        self.pinn.zero_grad()
        
        def closure():
            loss, loss_res, loss_bc, loss_init = solver.compute_loss(self.pinn, res_points, bc_points, init_points)
            loss.backward()
            return loss.item()
        
        loss = closure()
        
        # Compute Hessian-vector product
        grad = [param.grad.data.clone() for param in self.pinn.parameters()]
        
        # Use Nyström approximation for Hessian
        if self.H is None:
            self.H = torch.eye(len(grad)) * 1.0
            self.H_inv = torch.eye(len(grad)) * 1.0
        
        # Compute Hessian approximation
        H = self.H
        H_inv = self.H_inv
        
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
            else:
                p = grad
            # Compute Hessian-vector product
            h = [torch.zeros_like(p_i) for p_i in p]
            for i in range(len(p)):
                for j in range(len(p)):
                    h[i] += H[i][j] * p[j]
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
        for param, grad_i in zip(self.pinn.parameters(), grad):
            param.data = param.data - self.learning_rate * grad_i
        
        return loss.item()

def main():
    """Main function to reproduce results from paper"""
    parser = argparse.ArgumentParser(description='Reproduce PINN results')
    parser.add_argument('--pde', type=str, default='wave', help='PDE type (convection, reaction, wave)')
    parser.add_argument('--epochs', type=int, default=41000, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize PDE solver
    solver = PDESolver(pde_type=args.pde)
    
    # Initialize PINN
    pinn = PINN(input_dim=2, hidden_layers=[50, 50, 50], output_dim=1)
    
    # Initialize optimizers
    adam = AdamOptimizer(pinn, learning_rate=args.learning_rate)
    lbfgs = LBFGSOptimizer(pinn, learning_rate=1.0, history_size=10)
    nysncg = NysNewtonCG(pinn, learning_rate=0.01, max_iter=10)
    
    # Training loop
    losses = []
    for epoch in range(args.epochs):
        # Adam step
        loss = adam.step(solver.res_points, solver.bc_points, solver.bc_points, solver)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
        losses.append(loss)
    
    # L-BFGS step after Adam
    for epoch in range(1000):
        loss = lbfgs.step(solver.res_points, solver.bc_points, solver.bc_points, solver)
        if epoch % 1000 == 0:
            print(f"LBFGS Epoch {epoch}, Loss: {loss:.6f}")
        losses.append(loss)
    
    # NysNewton-CG step
    for epoch in range(1000):
        loss = nysncg.step(solver.res_points, solver.bc_points, solver.bc_points, solver)
        if epoch % 1000 == 0:
            print(f"NNCG Epoch {epoch}, Loss: {loss:.6f}")
        losses.append(loss)
    
    # Save results
    np.save(os.path.join(args.output_dir, 'losses.npy'), np.array(losses))
    
    # Plot results
    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(args.output_dir, 'loss_plot.png'))
        plt.show()
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()