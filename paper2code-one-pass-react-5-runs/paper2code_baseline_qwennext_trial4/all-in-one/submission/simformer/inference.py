"""
Inference for Simformer
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Inference engine for Simformer.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize inference engine.
        
        Args:
            model: Model to use for inference
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def infer(self, 
              inputs: torch.Tensor,
              condition_mask: torch.Tensor,
              num_samples: int = 1000,
            ) -> torch.Tensor:
        """
        Perform inference.
        
        Args:
            inputs: Input tensor
            condition_mask: Condition mask
            num_samples: Number of samples
        Returns:
            samples: Sampled tensor
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Sample from noise distribution
        noise = torch.randn(batch_size, seq_len, device=self.device)
        
        # Sample from diffusion model
        samples = self.model.sample(noise, condition_mask, num_steps=50)
        
        return samples
    
    def infer_with_guidance(self, 
                              inputs: torch.Tensor,
                              condition_mask: torch.Tensor,
                              guidance_fn: callable,
                              num_samples: int = 1000,
                            ) -> torch.Tensor:
        """
        Perform inference with guidance.
        
        Args:
            inputs: Input tensor
            condition_mask: Condition mask
            guidance_fn: Guidance function
            num_samples: Number of samples
        Returns:
            samples: Sampled tensor
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Sample from noise distribution
        noise = torch.randn(batch_size, seq_len, device=self.device)
        
        # Sample from diffusion model with guidance
        samples = self.model.sample(noise, condition_mask, guidance_fn, num_steps=50)
        
        return samples
    
    def sample_conditionals(self, 
                           inputs: torch.Tensor,
                           condition_mask: torch.Tensor,
                           num_samples: int = 1000,
                           conditionals: List[str] = ["posterior", "likelihood", "joint"]) -> dict:
        """
        Sample from conditionals of the joint distribution.
        
        Args:
            inputs: Input tensor
            condition_mask: Condition mask
            num_samples: Number of samples
            conditionals: List of conditionals to sample from
        Returns:
            samples: Dictionary of samples for each conditional
        """
        samples = {}
        
        for conditional in conditionals:
            if conditional == "posterior":
                # Sample from posterior
                posterior_samples = self.infer(inputs, condition_mask, num_samples)
                samples["posterior"] = posterior_samples
            elif conditional == "likelihood":
                # Sample from likelihood
                likelihood_samples = self.infer(inputs, condition_mask, num_samples)
                samples["likelihood"] = likelihood_samples
            elif conditional == "joint":
                # Sample from joint
                joint_samples = self.infer(inputs, condition_mask, num_samples)
                samples["joint"] = joint_samples
            elif conditional == "prior":
                # Sample from prior
                prior_samples = self.infer(inputs, condition_mask, num_samples)
                samples["prior"] = prior_samples
        
        return samples