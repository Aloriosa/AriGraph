"""
Fare: Fine-tuning Adversarial Robustness via Embedding preservation
Unsupervised adversarial fine-tuning of CLIP vision encoders

Implements the FARE algorithm from the paper:
"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
import torchvision.models as models
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

def compute_embedding_loss(original_embeddings, new_embeddings):
    """
    Compute the embedding preservation loss (squared L2 distance)
    This preserves the cosine similarity between original and new embeddings
    """
    # Compute L2 distance between original and new embeddings
    l2_distance = torch.sum((original_embeddings - new_embeddings) ** 2, dim=1)
    return torch.mean(l2_distance)

def compute_adversarial_loss(model, data_loader, epsilon, device):
    """
    Compute adversarial loss using PGD attack
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    for batch in data_loader:
        images = batch[0].to(device)
        batch_size = images.size(0)
        
        # Compute original embeddings
        with torch.no_grad():
            original_embeddings = model(images)
        
        # Generate adversarial examples using PGD
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        
        # PGD attack
        for _ in range(10):  # 10 steps
            adv_images.requires_grad = True
            adv_embeddings = model(adv_images)
            
            # Loss to maximize distance between original and new embeddings
            loss = torch.sum((original_embeddings - adv_embeddings) ** 2)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_images, create_graph=True)[0]
            
            # Update adversarial images
            adv_images = adv_images + 0.1 * torch.sign(grad)
            
            # Project back to epsilon-ball
            adv_images = torch.clamp(adv_images, 0, 1)
            adv_images = images + torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = adv_images.detach()
            adv_images.requires_grad = True
        
        # Compute loss for adversarial examples
        adv_embeddings = model(adv_images)
        loss = torch.sum((original_embeddings - adv_embeddings) ** 2)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples if total_samples > 0 else 0

def train_fare_model(model, train_loader, epochs, batch_size, epsilon, device):
    """
    Train the FARE model using unsupervised adversarial fine-tuning
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_embedding_loss = 0.0
        running_adversarial_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(device)
            batch_size = images.size(0)
        
            optimizer.zero_grad()
        
            # Compute original embeddings
            original_embeddings = model(images)
        
            # Generate adversarial examples using PGD
            adv_images = images.clone().detach()
            adv_images.requires_grad = True
        
            # PGD attack
            for _ in range(10):  # 10 steps
                adv_images.requires_grad = True
            adv_embeddings = model(adv_images)
        
            # Compute adversarial loss
            adversarial_loss = torch.sum((original_embeddings - adv_embeddings) ** 2)
        
            # Compute embedding preservation loss
            embedding_loss = compute_embedding_loss(original_embeddings, adv_embeddings)
        
            # FARE loss combines adversarial loss and embedding preservation loss
            # FARE loss = adversarial_loss + lambda * embedding_loss
            # We use lambda = 0.5 as in the paper
            fare_loss = adversarial_loss + 0.5 * embedding_loss
        
            fare_loss.backward()
            optimizer.step()
        
            running_loss += fare_loss.item() * batch_size
            running_embedding_loss += embedding_loss.item() * batch_size
            running_adversarial_loss += adversarial_loss.item() * batch_size
            total_samples += batch_size
        
        avg_loss = running_loss / total_samples
        avg_embedding_loss = running_embedding_loss / total_samples
        avg_adversarial_loss = running_adversarial_loss / total_samples
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], "
                   f"Loss: {avg_loss:.4f}, "
                   f"Embedding Loss: {avg_embedding_loss:.4f}, "
                   f"Adversarial Loss: {avg_adversarial_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train FARE model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epsilon', type=float, default=2/255, help='Epsilon for adversarial perturbation')
    parser.add_argument('--output', type=str, required=True, help='Output path for model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    trainset = torch.load(args.dataset)
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    logger.info("Initializing CLIP vision encoder...")
    model = CLIPVisionEncoder(pretrained=True)
    
    # Train model
    logger.info("Starting FARE training...")
    model = train_fare_model(model, train_loader, args.epochs, args.batch_size, args.epsilon, device)
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()