import torch
import numpy as np
from typing import List, Dict, Any
from src.data.simulator_wrapper import SimulatorWrapper

class Tokenizer:
    def __init__(self, 
                 vocab_size: int = 1000, 
                 embedding_dim: int = 128, 
                 fourier_dim: int = 64,
                 max_freq: float = 10.0):
        """
        Initialize the Tokenizer with learnable embeddings for variable identifiers and condition states,
        and support for Random Fourier Features for space/time-dependent parameters.
        
        Args:
            vocab_size: Number of unique variable identifiers
            embedding_dim: Dimension of learnable embeddings for identifiers and condition states
            fourier_dim: Dimension of Random Fourier embeddings for space/time parameters
            max_freq: Maximum frequency for Fourier features
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.fourier_dim = fourier_dim
        self.max_freq = max_freq
        
        # Learnable embeddings for variable identifiers (0 to vocab_size-1)
        self.id_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Learnable embeddings for conditional states: observed (0) and latent (1)
        self.condition_embedding = torch.nn.Embedding(2, embedding_dim)
        
        # Random Fourier feature matrix (fixed, not learned)
        self.fourier_matrix = torch.randn(fourier_dim, embedding_dim) * max_freq
        
        # Register as buffer so it moves with the model but isn't trained
        self.register_buffer('fourier_matrix_buffer', self.fourier_matrix)
        
        # Track variable names for decoding
        self.id_to_name = {i: f"var_{i}" for i in range(vocab_size)}
        self.name_to_id = {f"var_{i}": i for i in range(vocab_size)}
    
    def encode(self, sim_output: dict) -> List[Dict]:
        """
        Convert simulator output into a sequence of tokens.
        Each token is a dict containing:
        - 'id': variable identifier (int)
        - 'value': variable value (float or tensor)
        - 'condition': conditional state (0=observed, 1=latent)
        - 'embedding': learned embedding vector (tensor of shape [embedding_dim])
        
        For space/time-dependent parameters, use Random Fourier embeddings.
        """
        tokens = []
        
        # Extract parameters and data from simulator output
        # According to paper context: parameters_or_data_may_be_functions_of_space_or_time
        parameters = sim_output.get('parameters', {})
        data = sim_output.get('data', {})
        
        # Process parameters (assumed to be space/time-dependent)
        for var_name, var_value in parameters.items():
            if isinstance(var_value, (int, float)):
                var_value = torch.tensor(float(var_value))
            elif isinstance(var_value, np.ndarray):
                var_value = torch.from_numpy(var_value).float()
            
            # Get variable id
            var_id = self.name_to_id.get(var_name, 0)  # fallback to 0 if unknown
            
            # Determine condition: parameters are typically latent
            condition = 1  # latent
            
            # Create embedding for identifier
            id_emb = self.id_embedding(torch.tensor(var_id))
            
            # Create embedding for condition
            cond_emb = self.condition_embedding(torch.tensor(condition))
            
            # Apply Random Fourier embedding for space/time-dependent parameters
            # Convert scalar value to 1D tensor for Fourier encoding
            if isinstance(var_value, torch.Tensor) and var_value.numel() == 1:
                x = var_value.item()
                fourier_features = self._random_fourier_features(x)
            else:
                # For non-scalar values, use mean or flatten
                if isinstance(var_value, torch.Tensor):
                    x = var_value.mean().item() if var_value.numel() > 0 else 0.0
                else:
                    x = float(var_value)
                fourier_features = self._random_fourier_features(x)
            
            # Combine embeddings: identifier + condition + Fourier features
            # We'll concatenate and project back to embedding_dim if needed
            combined_emb = id_emb + cond_emb + torch.tensor(fourier_features, dtype=torch.float32)
            
            tokens.append({
                'id': var_id,
                'value': var_value,
                'condition': condition,
                'embedding': combined_emb.detach().cpu().numpy()
            })
        
        # Process data (assumed to be observed)
        for var_name, var_value in data.items():
            if isinstance(var_value, (int, float)):
                var_value = torch.tensor(float(var_value))
            elif isinstance(var_value, np.ndarray):
                var_value = torch.from_numpy(var_value).float()
            
            # Get variable id
            var_id = self.name_to_id.get(var_name, 0)
            
            # Data is observed
            condition = 0  # observed
            
            # Create embeddings
            id_emb = self.id_embedding(torch.tensor(var_id))
            cond_emb = self.condition_embedding(torch.tensor(condition))
            
            # Apply Fourier features
            if isinstance(var_value, torch.Tensor) and var_value.numel() == 1:
                x = var_value.item()
                fourier_features = self._random_fourier_features(x)
            else:
                if isinstance(var_value, torch.Tensor):
                    x = var_value.mean().item() if var_value.numel() > 0 else 0.0
                else:
                    x = float(var_value)
                fourier_features = self._random_fourier_features(x)
            
            combined_emb = id_emb + cond_emb + torch.tensor(fourier_features, dtype=torch.float32)
            
            tokens.append({
                'id': var_id,
                'value': var_value,
                'condition': condition,
                'embedding': combined_emb.detach().cpu().numpy()
            })
        
        return tokens
    
    def decode(self, tokens: List[Dict]) -> Dict:
        """
        Convert token sequence back to simulator output format.
        Returns a dict with 'parameters' and 'data' keys.
        """
        parameters = {}
        data = {}
        
        for token in tokens:
            var_id = token['id']
            value = token['value']
            condition = token['condition']
            var_name = self.id_to_name.get(var_id, f"var_{var_id}")
            
            if condition == 1:  # latent (parameter)
                parameters[var_name] = value
            else:  # observed (data)
                data[var_name] = value
        
        return {
            'parameters': parameters,
            'data': data
        }
    
    def _random_fourier_features(self, x: float) -> np.ndarray:
        """
        Generate Random Fourier Features for a scalar input x.
        Uses: cos(ω·x + b) and sin(ω·x + b) where ω is from fourier_matrix and b is random phase.
        """
        # Convert x to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1)
        
        # Generate random phase for each frequency
        phase = torch.rand(self.fourier_dim) * 2 * np.pi
        
        # Compute Fourier features: cos(ω·x + b)
        # fourier_matrix: [fourier_dim, embedding_dim]
        # x_tensor: [1, 1]
        # We need to project x to the fourier_dim space
        omega = self.fourier_matrix_buffer  # [fourier_dim, embedding_dim]
        
        # For simplicity, we treat each row of omega as a frequency vector for one dimension
        # We'll use the first column of omega as the frequency for our scalar x
        # This is a simplification - in practice, we might have multi-dimensional inputs
        freqs = omega[:, 0]  # [fourier_dim]
        
        # Compute cos(ω·x + b)
        fourier_features = torch.cos(freqs * x_tensor.squeeze() + phase)
        
        # Return as numpy array
        return fourier_features.detach().cpu().numpy()
    
    def to(self, device):
        """Move embeddings to specified device."""
        self.id_embedding = self.id_embedding.to(device)
        self.condition_embedding = self.condition_embedding.to(device)
        self.fourier_matrix_buffer = self.fourier_matrix_buffer.to(device)
        return self