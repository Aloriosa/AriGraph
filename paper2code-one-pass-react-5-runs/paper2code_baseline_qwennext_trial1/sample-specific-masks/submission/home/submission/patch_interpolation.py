"""
Patch-wise Interpolation Module for SMM framework
This implements the patch-wise interpolation module described in Section 3.3 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchWiseInterpolation(nn.Module):
    """
    Patch-wise interpolation module.
    This module upscales a mask from a smaller size to the original size.
    """
    def __init__(self, patch_size=8):
        """
        Initialize the patch-wise interpolation module.
        Args:
            patch_size (int): Size of patches (default: 8)
        """
        super(PatchWiseInterpolation, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, H//l, W//l) or (batch_size, 3, H//l, W//l)
        Returns:
            Tensor: Output tensor of shape (batch_size, 1, H, W) or (batch_size, 3, H, W)
        """
        # Get input shape
        batch_size, channels, height, width = x.shape
        
        # Calculate scale factor
        scale_factor = self.patch_size
        
        # Use nearest neighbor interpolation for simplicity
        # This matches the paper's description of "assigning the same value to surrounding areas"
        x = F.interpolate(
            x, 
            scale_factor=scale_factor, 
            mode='nearest'
        )
        
        return x

# Test function
def test_patch_interpolation():
    """
    Test the patch-wise interpolation module
    """
    print("Testing PatchWiseInterpolation...")
    model = PatchWiseInterpolation(patch_size=8)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("Test passed!")

if __name__ == '__main__':
    test_patch_interpolation()