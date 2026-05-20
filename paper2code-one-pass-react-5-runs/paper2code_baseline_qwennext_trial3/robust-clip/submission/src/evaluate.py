"""
Evaluate the robustness of the FARE model on test data
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPVisionEncoder(nn.Module):
    """
    Simplified CLIP vision encoder based on ViT-L/14 architecture
    """
    def __init__(self, pretrained=True):
        super(CLIPVisionEncoder, self).__init__()
        # Use a pre-trained Vision Transformer
        self.vit = models.vit_l_14(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove the classification head
        self.vit.heads = nn.Identity()
        
        # Add a small projection head for contrastive learning
        self.projection = nn.Linear(1024, 512)
        
    def forward(self, x):
        # Pass through ViT
        x = self.vit(x)
        # Project to final embedding space
        x = self.projection(x)
        # Normalize embeddings
        x = nn.functional.normalize(x, dim=-1)
        return x

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    """
    model.to(device)
    model.eval()
    
    # Evaluate clean accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
        labels = batch[1].to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    clean_accuracy = 100 * correct / total
    
    # Evaluate robust accuracy using PGD attack
    robust_correct = 0
    total = 0
    for batch in test_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # Generate adversarial examples using PGD
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        
        # PGD attack
        for _ in range(10):  # 10 steps
            adv_images.requires_grad = True
            adv_outputs = model(adv_images)
            
            # Loss to maximize distance between original and new embeddings
            loss = torch.sum((adv_outputs - model(images)) ** 2)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_images, create_graph=True)[0]
            
            # Update adversarial images
            adv_images = adv_images + 0.1 * torch.sign(grad)
            
            # Project back to epsilon-ball
            adv_images = torch.clamp(adv_images, 0, 1)
            adv_images = images + torch.clamp(adv_images - images, -2/255, 2/255)
            adv_images = adv_images.detach()
            adv_images.requires_grad = True
        
        # Evaluate on adversarial examples
        adv_outputs = model(adv_images)
        _, predicted = torch.max(adv_outputs.data, 1)
        total += labels.size(0)
        robust_correct += (predicted == labels).sum().item()
    
    robust_accuracy = 100 * robust_correct / total if total > 0 else 0
    
    return {
        "clean_accuracy": clean_accuracy,
        "robust_accuracy": robust_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate FARE model')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output', type=str, required=True, help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    testset = torch.load(args.dataset)
    
    # Create data loaders
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # Initialize model
    logger.info("Initializing model...")
    model = CLIPVisionEncoder(pretrained=False)
    model.load_state_dict(torch.load(args.model))
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()