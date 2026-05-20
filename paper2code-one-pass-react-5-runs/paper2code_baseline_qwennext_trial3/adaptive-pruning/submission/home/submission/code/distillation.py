#!/usr/bin/env python3
"""
Self-knowledge distillation for APT
Implements the efficient self-distillation technique from Section 4.4
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

class SelfDistillation:
    """
    Self-knowledge distillation for APT
    Uses the model itself as both teacher and student to recover performance
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 distill_ratio: float = 0.5,
                 distill_start_step: int = 0,
                 distill_end_step: int = 1000):
        """
        Initialize self-distillation
        
        Args:
            model: The model to apply distillation to
            distill_ratio: Ratio of distillation loss to task loss
            distill_start_step: Step to start distillation
            distill_end_step: Step to end distillation
        """
        self.model = model
        self.distill_ratio = distill_ratio
        self.distill_start_step = distill_start_step
        self.distill_end_step = distill_end_step
        self.current_step = 0
        self.teacher_model = None
        self.layer_mapping = {}
        
        # Create a copy of the model for teacher
        self._create_teacher_model()
        
    def _create_teacher_model(self):
        """Create teacher model as a copy of the student"""
        self.teacher_model = type(self.model).from_pretrained(self.model.config.name_or_path)
        self.teacher_model.load_state_dict(self.model.state_dict())
        self.teacher_model.eval()  # Teacher is always in eval mode
        
    def get_distill_weight(self, step: int) -> float:
        """
        Get distillation weight based on step (linearly increases from 0 to 1)
        """
        if step < self.distill_start_step:
            return 0.0
        elif step > self.distill_end_step:
            return 1.0
        else:
            return (step - self.distill_start_step) / (self.distill_end_step - self.distill_start_step)
            
    def compute_distillation_loss(self, 
                                 student_outputs: torch.Tensor,
                                 teacher_outputs: torch.Tensor,
                                 student_hidden_states: List[torch.Tensor],
                                 teacher_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher
        Implements the MSE loss on hidden states as described in paper
        """
        if self.teacher_model is None:
            return torch.tensor(0.0)
            
        # Layer-wise MSE loss on hidden states
        distill_loss = torch.tensor(0.0)
        
        # Sample teacher layers randomly (as in paper)
        num_layers = len(student_hidden_states)
        sampled_indices = torch.randperm(num_layers)[:max(1, num_layers // 2)]
        
        for i in sampled_indices:
            if i < len(teacher_hidden_states):
                # Apply layer transformation as described in paper
                # In practice, we use a simple identity transformation
                student_hidden = student_hidden_states[i]
                teacher_hidden = teacher_hidden_states[i]
                
                # Compute MSE loss
                loss = torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
                distill_loss += loss
                
        return distill_loss / len(sampled_indices) if len(sampled_indices) > 0 else torch.tensor(0.0)
        
    def update_teacher(self):
        """Update teacher model with current student weights"""
        if self.teacher_model is not None:
            self.teacher_model.load_state_dict(self.model.state_dict())
            
    def forward(self, 
                student_input: torch.Tensor,
                teacher_input: torch.Tensor,
                student_outputs: torch.Tensor,
                student_hidden_states: List[torch.Tensor],
                teacher_outputs: torch.Tensor,
                teacher_hidden_states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with distillation
        Returns: (task_loss, distill_loss)
        """
        # Update teacher periodically
        if self.current_step % 10 == 0:
            self.update_teacher()
            
        # Compute task loss (cross entropy, etc.)
        task_loss = torch.nn.functional.cross_entropy(student_outputs, teacher_outputs.argmax(dim=-1))
        
        # Compute distillation loss
        distill_loss = self.compute_distillation_loss(
            student_outputs, teacher_outputs, student_hidden_states, teacher_hidden_states
        )
        
        # Combine losses
        distill_weight = self.get_distill_weight(self.current_step)
        total_loss = (1 - distill_weight) * task_loss + distill_weight * distill_loss
        
        self.current_step += 1
        
        return total_loss, distill_loss