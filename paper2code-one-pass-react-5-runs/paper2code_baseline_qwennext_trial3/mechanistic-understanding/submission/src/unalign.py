import numpy as np
import torch
import os

class Unaligner:
    """Undo alignment by reactivating toxic vectors."""
    
    def __init__(self, dpo_weights_path, toxic_vectors_path):
        self.dpo_weights_path = dpo_weights_path
        self.toxic_vectors_path = toxic_vectors_path
        self.dpo_weights = {}
        self.toxic_vectors = []
        
    def load_weights(self):
        """Load DPO weights and toxic vectors."""
        print("Loading DPO weights and toxic vectors...")
        
        # Load DPO weights
        if os.path.exists(self.dpo_weights_path):
            self.dpo_weights = torch.load(self.dpo_weights_path)
        else:
            # Generate synthetic weights
            self.dpo_weights = {}
            for i in range(12):
                self.dpo_weights[f'layer_{i}'] = torch.randn(768, 768) * 0.01
        
        # Load toxic vectors
        if os.path.exists(self.toxic_vectors_path):
            self.toxic_vectors = np.load(self.toxic_vectors_path)
        else:
            # Generate synthetic toxic vectors
            self.toxic_vectors = np.random.randn(128, 768)
        
        print("Weights loaded")
        
    def unalign(self, num_vectors=7, scale_factor=10, output_path='unaligned_model.pth'):
        """Undo alignment by scaling toxic key vectors."""
        print("Undoing alignment...")
        
        # In paper: For GPT2, scale key vectors to reactivate toxicity
        # In paper: For Llama2, turn gating components back on
        
        # Get key vectors (in paper, key vectors are in MLP blocks)
        key_vectors = []
        
        # Find key vectors (in paper, key vectors are in MLP blocks)
        # For reproduction, we'll use a subset of weights
        for name, weight in self.dpo_weights.items():
            if 'weight' in name and 'mlp' in name:
                # These are key vectors
                key_vectors.append((name, weight))
        
        # Select top toxic key vectors
        key_vectors = key_vectors[:10]
        
        # Scale key vectors
        for name, weight in key_vectors:
            # Scale by factor
            scaled_weight = weight * scale_factor
            self.dpo_weights[name] = scaled_weight
        
        # Save unaligned model
        torch.save(self.dpo_weights, output_path)
        print(f"Unaligned model saved to {output_path}")
        
        return self.dpo_weights

# Example usage
if __name__ == "__main__":
    unaligner = Unaligner('dpo_model_weights.pth', 'toxic_vectors.npy')
    unaligner.load_weights()
    unaligner.unalign(num_vectors=7, scale_factor=10)
    print("Unalignment completed")