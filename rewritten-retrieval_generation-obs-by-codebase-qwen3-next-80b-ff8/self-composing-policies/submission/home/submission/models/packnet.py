#!/usr/bin/env python3
"""
PackNet implementation for baseline comparison.
This architecture prunes weights after training on each task and reuses them for new tasks.
"""
import os
import torch
import torch.nn as nn
import copy
from typing import List

class PackNet(nn.Module):
    """
    PackNet implementation that prunes weights after training on each task
    and reuses them for new tasks.
    """
    def __init__(
        self,
        hidden_dim,
        task_id,
        total_task_num,
        is_first_task,
        layer_init=lambda x, **kwargs: x,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        assert task_id > 0, "Task ID must be greater than 0 in PackNet"
        
        # Simple encoder network
        self.model = nn.Sequential(
            layer_init(nn.Linear(24, hidden_dim)),  # Assuming 24-dim state space
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        
        self.task_id = task_id
        self.total_task_num = total_task_num
        self.prune_percentage = 1 / self.total_task_num
        
        self.view = None
        self.handled_layers = []
        
        # Generate masks for each task
        self.masks = []
        for name, param in self.model.named_parameters():
            if name.endswith(".weight"):
                self.masks.append(
                    torch.zeros(param.size(), dtype=torch.long, device=device)
                )
            else:
                self.masks.append(None)
        
        # Freeze biases for non-first tasks
        if not is_first_task:
            for name, param in self.model.named_parameters():
                if name.endswith(".bias"):
                    param.requires_grad = False
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)  # Assuming 1-dim action space
        self.logstd_head = nn.Linear(hidden_dim, 1)
        
    def adjust_gradients(self, retrain_mode=False):
        """Adjust gradients based on masks."""
        mask_id = self.task_id if retrain_mode else 0
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue
            p.grad = p.grad * (mask == mask_id)
    
    @torch.no_grad()
    def prune(self):
        """Prune weights for current task."""
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue

            # Sort the unassigned weights from lower to higher magnitudes
            masked = p * (mask == 0)  # only select "free" weights
            flat = masked.flatten()
            _sorted, indices = torch.sort(flat.abs(), descending=True)  # sort from max to min magnitude
            n_prune = int(self.prune_percentage * flat.size(0))  # number of weights to keep in pruning

            # Create the mask
            mask.flatten()[indices[:n_prune]] = self.task_id
    
    def forward(self, x):
        """Forward pass with mask applied."""
        x = self.model(x)
        mean = self.mean_head(x)
        log_std = self.logstd_head(x)
        return mean, log_std
    
    @torch.no_grad()
    def set_view(self, task_id):
        """Set which task's weights to use."""
        if task_id is None and self.view is not None:
            # Restore the original state of the model in the free parameters (not masked)
            for param_copy, param, mask in zip(
                self.handled_layers, self.model.parameters(), self.masks
            ):
                if param_copy is None:
                    continue
                m = torch.logical_and(mask <= self.view, mask > 0)  # pruned=0, not-pruned=1
                param.data += param_copy.data * torch.logical_not(m)

            self.handled_layers = []
            self.view = task_id
            return

        if len(self.handled_layers) == 0:
            # Save a copy of each (parametrized) layer of the model
            for param, mask in zip(self.model.parameters(), self.masks):
                if mask is not None:
                    self.handled_layers.append(copy.deepcopy(param))
                else:
                    self.handled_layers.append(None)

        # Apply the masks
        for p, mask in zip(self.model.parameters(), self.masks):
            if mask is None:
                continue
            # Set to zero the parameters that are free (have no mask) or whose mask ID is greater than task_id
            p.data *= torch.logical_and(mask <= task_id, mask > 0)

        self.view = task_id
    
    def save(self, dirname):
        """Save the model."""
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")
    
    @staticmethod
    def load(dirname, map_location=None):
        """Load the model."""
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        return model