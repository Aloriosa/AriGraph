"""
Model utilities for SMM framework
This module provides utilities for loading and preprocessing models.
"""
import torch
import torchvision.models as models

def load_model(model_name, num_classes):
    """
    Load a pre-trained model from torchvision
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit_b32":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# Test function
def test_models():
    """
    Test the model loading utilities
    """
    print("Testing models utilities...")
    models = ["resnet18", "resnet50", "vit_b32"]
    num_classes = 10
    
    for model_name in models:
        print(f"Loading model: {model_name}")
        model = load_model(model_name, num_classes)
        print(f"Model: {model}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("Test passed!")

if __name__ == '__main__':
    test_models()