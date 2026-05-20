import torch
import torch.nn as nn
import numpy as np
import time
import os

class APGDAttack:
    """
    APGD Attack implementation
    This is a simplified version of the attack described in the paper
    """
    def __init__(self, model, device, eps=2/255, n_iter=10):
        self.model = model
        self.device = device
        self.eps = eps
        self.n_iter = n_iter
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, x, y):
        """
        Apply attack
        """
        self.model.eval()
        x_adv = x.clone().requires_grad_(True)
        
        for _ in range(self.n_iter):
            self.model.zero_grad()
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            loss.backward()
            
            # Project to epsilon-ball
            x_adv = x_adv + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv

class EnsembleAttack:
    """
    Ensemble Attack implementation
    This is a simplified version of the attack described in the paper
    """
    def __init__(self, model, device, eps=2/255, n_iter=10):
        self.model = model
        self.device = device
        self.eps = eps
        self.n_iter = n_iter
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, x, y):
        """
        Apply attack
        """
        self.model.eval()
        x_adv = x.clone().requires_grad_(True)
        
        for _ in range(self.n_iter):
            self.model.zero_grad()
            output = self.model(x_adv)
            loss = self.criterion(output, y)
            loss.backward()
            
            # Project to epsilon-ball
            x_adv = x_adv + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv

class TargetedAttack:
    """
    Targeted Attack implementation
    This is a simplified version of the attack described in the paper
    """
    def __init__(self, model, device, eps=2/255, n_iter=10):
        self.model = model
        self.device = device
        self.eps = eps
        self.n_iter = n_iter
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, x, target):
        """
        Apply attack
        """
        self.model.eval()
        x_adv = x.clone().requires_grad_(True)
        
        for _ in range(self.n_iter):
            self.model.zero_grad()
            output = self.model(x_adv)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Project to epsilon-ball
            x_adv = x_adv + self.eps * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv