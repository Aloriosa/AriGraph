import torch
from src.data.simulator_wrapper import SimulatorWrapper

class AttentionMaskBuilder:
    def __init__(self, simulator_wrapper: SimulatorWrapper):
        self.simulator_wrapper = simulator_wrapper

    def build_attention_mask(self, sim_metadata: dict) -> torch.Tensor:
        """
        Constructs a dynamic attention mask based on simulator dependency structure.
        Encodes conditional independencies to guide transformer attention.
        
        Args:
            sim_metadata: Dictionary containing simulator metadata including
                         dependency structure information.
                         
        Returns:
            torch.Tensor: Attention mask of shape (batch_size, seq_len, seq_len)
                         where 0 means attention is allowed and -inf means masked.
        """
        batch_size = sim_metadata.get('batch_size', 1)
        seq_len = sim_metadata.get('seq_len', 1)
        
        # Initialize mask with zeros (allow all attention)
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Extract dependency structure from simulator metadata
        # This assumes simulator_metadata contains a 'dependency_matrix' or similar
        dependency_matrix = sim_metadata.get('dependency_matrix', None)
        
        if dependency_matrix is not None:
            # Convert dependency matrix to attention mask
            # dependency_matrix[i][j] = 1 means i depends on j, so j can attend to i
            # We want to mask positions where there is no dependency
            dependency_tensor = torch.tensor(dependency_matrix, dtype=torch.float32)
            
            # Expand to match batch dimension
            if dependency_tensor.dim() == 2:
                dependency_tensor = dependency_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Create mask: 0 where dependency exists, -inf where it doesn't
            mask = torch.where(dependency_tensor == 1, mask, torch.full_like(mask, float('-inf')))
        else:
            # Fallback: if no dependency structure provided, use full attention
            # This is the default case for generic simulation
            pass
            
        return mask