"""
Mask Generator implementation for SMM framework
This implements the lightweight CNN for generating sample-specific masks
as described in Section 3.2 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """
    Lightweight mask generator based on CNN architecture.
    This is a 5-layer CNN that takes a resized image and outputs a 3-channel mask.
    """
    def __init__(self, in_channels=3, num_layers=5, base_channels=32):
        """
        Initialize the mask generator.
        Args:
            in_channels (int): Number of input channels (default: 3 for RGB)
            num_layers (int): Number of convolutional layers (default: 5)
            base_channels (int): Base number of channels (default: 32)
        """
        super(MaskGenerator, self).__init__()
        
        # Define layers
        layers = []
        
        # First layer: input to base_channels
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True)
        
        # Middle layers: base_channels to base_channels
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
            layers.append(nn.BatchNorm2d(base_channels))
            layers.append(nn.ReLU(inplace=True)
        
        # Last layer: base_channels to 3 channels for output mask
        layers.append(nn.Conv2d(base_channels, 3, kernel_size=3, padding=1, bias=False)
        
        # Final activation: Sigmoid to ensure values between 0 and 1
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (as suggested in the paper, we use a standard initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, H, W)
        Returns:
            Tensor: Output mask of shape (batch_size, 3, H, W)
        """
        return self.network(x)

# Test function
def test_mask_generator():
    """
    Test the mask generator
    """
    print("Testing MaskGenerator...")
    model = MaskGenerator(in_channels=3, num_layers=5, base_channels=32)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("Test passed!")

if __name__ == '__main__':
    test_mask_generator()