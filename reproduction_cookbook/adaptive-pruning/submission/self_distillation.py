import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class SelfKnowledgeDistillation:
    """
    Self-knowledge distillation technique as described in the APT paper.
    Shares frozen parameters between student and teacher models to recover performance.
    """
    
    def __init__(self, student_model: nn.Module, teacher_model: Optional[nn.Module] = None,
                 device: torch.device = None):
        self.student_model = student_model
        self.teacher_model = teacher_model if teacher_model else student_model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distillation parameters
        self.distillation_weight = 0.0
        self.distillation_temperature = 2.0
        self.layer_mapping_strategy = "closest_nonpruned"  # as in paper
        self.distillation_loss_type = "mse"  # mean squared error
        
        # Store layer mappings
        self.layer_mappings = {}
        
        # Initialize layer mapping
        self._initialize_layer_mappings()
        
    def _initialize_layer_mappings(self):
        """Initialize mappings between student and teacher layers"""
        # Map layers by name and structure
        student_layers = {}
        teacher_layers = {}
        
        # Collect all layers
        for name, module in self.student_model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                student_layers[name] = module
                
        for name, module in self.teacher_model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                teacher_layers[name] = module
                
        # Create mappings based on layer type and position
        for student_name, student_layer in student_layers.items():
            # Find closest matching teacher layer
            if student_name in teacher_layers:
                # Direct match
                self.layer_mappings[student_name] = teacher_layers[student_name]
            else:
                # Find closest layer by type and position
                # For simplicity, we'll use the same layer name if it exists
                # In practice, this would be more sophisticated
                matching_teacher = None
                for teacher_name, teacher_layer in teacher_layers.items():
                    if (type(student_layer) == type(teacher_layer) and 
                        student_name.split('.')[-1] == teacher_name.split('.')[-1]):
                        matching_teacher = teacher_layer
                        break
                        
                if matching_teacher:
                    self.layer_mappings[student_name] = matching_teacher
                else:
                    # Use the first matching type
                    for teacher_name, teacher_layer in teacher_layers.items():
                        if type(student_layer) == type(teacher_layer):
                            self.layer_mappings[student_name] = teacher_layer
                            break
                            
    def compute_distillation_loss(self, student_outputs, teacher_outputs, 
                                layer_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher models.
        Uses mean squared error between hidden states as in the paper.
        """
        if self.distillation_loss_type == "mse":
            return self._compute_mse_loss(student_outputs, teacher_outputs, layer_outputs)
        elif self.distillation_loss_type == "kl":
            return self._compute_kl_loss(student_outputs, teacher_outputs)
        else:
            return torch.tensor(0.0, device=self.device)
            
    def _compute_mse_loss(self, student_outputs, teacher_outputs, 
                         layer_outputs: Optional[Dict] = None) -> torch.Tensor:
        """Compute mean squared error loss between hidden states"""
        if layer_outputs is None:
            # Use output layer
            if isinstance(student_outputs, tuple):
                student_hidden = student_outputs[0]
                teacher_hidden = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
            else:
                student_hidden = student_outputs
                teacher_hidden = teacher_outputs
                
            # Compute MSE between hidden states
            mse_loss = F.mse_loss(student_hidden, teacher_hidden)
            return mse_loss
            
        else:
            # Compute MSE between specific layers
            total_loss = torch.tensor(0.0, device=self.device)
            count = 0
            
            for student_layer_name, student_hidden in layer_outputs.items():
                if student_layer_name in self.layer_mappings:
                    teacher_layer = self.layer_mappings[student_layer_name]
                    # Get teacher hidden state
                    teacher_hidden = self._get_layer_output(teacher_layer, teacher_outputs)
                    
                    if teacher_hidden is not None and student_hidden is not None:
                        # Ensure same shape
                        if student_hidden.shape == teacher_hidden.shape:
                            mse_loss = F.mse_loss(student_hidden, teacher_hidden)
                            total_loss += mse_loss
                            count += 1
                            
            return total_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
            
    def _get_layer_output(self, layer: nn.Module, outputs) -> Optional[torch.Tensor]:
        """Extract hidden state from a specific layer"""
        if isinstance(outputs, tuple):
            # Try to find the layer output in the tuple
            for output in outputs:
                if isinstance(output, torch.Tensor) and output.shape[0] > 0:
                    return output
        elif isinstance(outputs, torch.Tensor):
            return outputs
            
        return None
        
    def _compute_kl_loss(self, student_outputs, teacher_outputs) -> torch.Tensor:
        """Compute KL divergence loss between output distributions"""
        # Apply temperature scaling
        student_logits = student_outputs / self.distillation_temperature
        teacher_logits = teacher_outputs / self.distillation_temperature
        
        # Apply softmax
        student_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        return kl_loss
        
    def update_distillation_weight(self, step: int, total_steps: int):
        """Update distillation weight using linear schedule"""
        self.distillation_weight = min(step / total_steps, 1.0)
        
    def apply_distillation(self, student_outputs, teacher_outputs, 
                          layer_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply self-distillation to student model outputs.
        Returns the combined loss.
        """
        # Compute distillation loss
        distillation_loss = self.compute_distillation_loss(student_outputs, teacher_outputs, layer_outputs)
        
        # Apply distillation weight
        weighted_distillation_loss = self.distillation_weight * distillation_loss
        
        return weighted_distillation_loss
        
    def get_distillation_loss(self, student_outputs, teacher_outputs, 
                             layer_outputs: Optional[Dict] = None) -> torch.Tensor:
        """Get the distillation loss without applying weight"""
        return self.compute_distillation_loss(student_outputs, teacher_outputs, layer_outputs)
        
    def set_teacher_model(self, teacher_model: nn.Module):
        """Set the teacher model"""
        self.teacher_model = teacher_model
        self._initialize_layer_mappings()
        
    def set_distillation_temperature(self, temperature: float):
        """Set the distillation temperature"""
        self.distillation_temperature = temperature
        
    def set_distillation_loss_type(self, loss_type: str):
        """Set the distillation loss type"""
        self.distillation_loss_type = loss_type