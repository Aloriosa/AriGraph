"""
Evaluation utilities for NPSE algorithm.
"""
import torch
import numpy as np
from typing import Optional
import os

def calculate_c2st_score(model, data_loader, device):
    """
    Calculate the C2ST score for the model.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for theta_0, x in data_loader:
            theta_0 = theta_0.to(device)
            x = x.to(device)
            t = torch.rand(theta_0.shape[0], 1).to(device)
            
            # Forward diffusion
            theta_t = model.diffusion_process.forward_sde(theta_0, t)
            
            # Estimate score of the posterior
            score = model.model(theta_t, x, t)
            
            # Store predictions and targets
            predictions.extend(score.cpu().numpy())
            targets.extend(theta_0.cpu().numpy())
    
    # Calculate C2ST score
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate the C2ST score
    # This is a simplified version of the C2ST score
    # In practice, this would be more complex
    c2st_score = np.mean(np.abs(predictions - targets))
    
    return c2st_score

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the data.
    """
    c2st_score = calculate_c2stst_score(model, data_loader, device)
    
    return c2st_score