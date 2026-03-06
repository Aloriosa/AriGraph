import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class OutlierAwareSalienceScorer:
    """
    Outlier-aware salience scoring function as described in the APT paper.
    Combines magnitude of weight-gradient products with kurtosis of activations.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store salience scores for each parameter block
        self.salience_scores = {}
        self.activation_kurtosis = {}
        self.gradient_magnitudes = {}
        
        # Register hooks to collect activations and gradients
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks to collect activations and gradients"""
        
        def forward_hook(module, input, output):
            """Collect activations during forward pass"""
            if isinstance(module, torch.nn.Linear):
                # Store input activations (for kurtosis calculation)
                if hasattr(module, 'input_activations'):
                    module.input_activations = input[0].detach().cpu()
                else:
                    module.input_activations = input[0].detach().cpu()
                    
        def backward_hook(module, grad_input, grad_output):
            """Collect gradients during backward pass"""
            if isinstance(module, torch.nn.Linear):
                # Store gradient magnitudes
                if hasattr(module, 'weight'):
                    weight_grad = module.weight.grad.detach().cpu()
                    if hasattr(module, 'weight_gradients'):
                        module.weight_gradients = weight_grad
                    else:
                        module.weight_gradients = weight_grad
                        
                # Store output gradients
                if len(grad_output) > 0 and grad_output[0] is not None:
                    output_grad = grad_output[0].detach().cpu()
                    if hasattr(module, 'output_gradients'):
                        module.output_gradients = output_grad
                    else:
                        module.output_gradients = output_grad
                        
        # Register hooks on all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Forward hook for activations
                hook = module.register_forward_hook(forward_hook)
                self.hooks.append(hook)
                
                # Backward hook for gradients
                hook = module.register_full_backward_hook(backward_hook)
                self.hooks.append(hook)
                
    def compute_salience_scores(self, dataloader, num_batches: int = 10) -> Dict[str, torch.Tensor]:
        """
        Compute outlier-aware salience scores for all parameter blocks.
        Uses the formula: salience = |weight * gradient| * kurtosis(activations)
        """
        self.model.eval()
        self.salience_scores = {}
        self.activation_kurtosis = {}
        self.gradient_magnitudes = {}
        
        # Collect activations and gradients over batches
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
                
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                
            # Backward pass
            if isinstance(outputs, torch.Tensor):
                loss = outputs.mean()
            else:
                loss = outputs.loss.mean() if hasattr(outputs, 'loss') else outputs[0].mean()
                
            loss.backward()
            
        # Compute salience scores for each parameter block
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Compute salience score for this module
                if hasattr(module, 'weight') and hasattr(module, 'weight_gradients'):
                    # Weight-gradient product
                    weight_grad_product = module.weight * module.weight_gradients
                    
                    # Compute magnitude of weight-gradient product
                    magnitude = torch.abs(weight_grad_product)
                    
                    # Compute kurtosis of activations
                    if hasattr(module, 'input_activations'):
                        # Flatten activations
                        activations = module.input_activations.view(-1, module.in_features)
                        
                        # Compute kurtosis for each input dimension
                        kurtosis_values = []
                        for i in range(activations.shape[1]):
                            if activations[:, i].numel() > 1:
                                # Compute kurtosis using scipy
                                kurt = self._compute_kurtosis(activations[:, i].numpy())
                                kurtosis_values.append(kurt)
                            else:
                                kurtosis_values.append(0.0)
                                
                        # Average kurtosis across input dimensions
                        avg_kurtosis = np.mean(kurtosis_values) if len(kurtosis_values) > 0 else 0.0
                    else:
                        avg_kurtosis = 1.0  # Default value if no activations available
                        
                    # Combine magnitude and kurtosis
                    # Use weighted sum: salience = magnitude * (1 + kurtosis)
                    salience = magnitude * (1 + avg_kurtosis)
                    
                    # Store salience scores
                    self.salience_scores[name] = salience.mean(dim=1) if len(salience.shape) > 1 else salience
                    self.activation_kurtosis[name] = avg_kurtosis
                    self.gradient_magnitudes[name] = torch.mean(torch.abs(module.weight_gradients))
                    
        # Clear gradients
        self.model.zero_grad()
        
        return self.salience_scores
        
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data using scipy-like implementation"""
        if len(data) < 2:
            return 0.0
            
        # Standardize data
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
            
        standardized = (data - mean) / std
        
        # Compute kurtosis (Fisher definition)
        kurt = np.mean(standardized ** 4) - 3.0
        
        return float(kurt)
        
    def get_block_salience(self, block_name: str) -> torch.Tensor:
        """Get salience scores for a specific block"""
        return self.salience_scores.get(block_name, torch.tensor(0.0))
        
    def get_layer_salience(self, layer_name: str) -> torch.Tensor:
        """Get salience scores for a specific layer"""
        # Look for blocks that match the layer name
        layer_salience = torch.tensor(0.0)
        for name, score in self.salience_scores.items():
            if layer_name in name:
                layer_salience = layer_salience + score.mean()
        return layer_salience
        
    def get_top_salient_blocks(self, num_blocks: int = 10) -> List[Tuple[str, float]]:
        """Get the top N most salient blocks"""
        # Sort blocks by salience score
        salience_list = [(name, score.mean().item()) for name, score in self.salience_scores.items()]
        salience_list.sort(key=lambda x: x[1], reverse=True)
        
        return salience_list[:num_blocks]
        
    def get_bottom_salient_blocks(self, num_blocks: int = 10) -> List[Tuple[str, float]]:
        """Get the bottom N least salient blocks"""
        # Sort blocks by salience score
        salience_list = [(name, score.mean().item()) for name, score in self.salience_scores.items()]
        salience_list.sort(key=lambda x: x[1])
        
        return salience_list[:num_blocks]
        
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []