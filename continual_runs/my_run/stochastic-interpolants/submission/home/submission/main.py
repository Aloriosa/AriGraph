import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from models import StochasticInterpolant
from data import get_dataloader
import torchvision.utils as vutils
import tqdm

def sample_time(batch_size, device):
    """Sample time steps uniformly from [0, 1]"""
    return torch.rand(batch_size, device=device)

def interpolate_sample(x0, x1, t):
    """Interpolate between x0 and x1 at time t"""
    # For stochastic interpolants: x_t = (1 - t) * x0 + t * x1
    return (1 - t) * x0 + t * x1

def compute_velocity(x0, x1, x_t, t):
    """Compute the velocity field: v(x_t, t) = x1 - x0"""
    return x1 - x0

def train_epoch(model, dataloader, optimizer, device, task, epoch, save_dir):
    model.train()
    total_loss = 0
    
    for batch_idx, (x0, x1, mask) in enumerate(dataloader):
        x0, x1, mask = x0.to(device), x1.to(device), mask.to(device)
        batch_size = x0.size(0)
        
        # Sample time
        t = sample_time(batch_size, device)
        t_expanded = t.view(-1, 1, 1, 1)
        
        # Create interpolated sample
        x_t = interpolate_sample(x0, x1, t_expanded)
        
        # Compute target velocity
        v_target = compute_velocity(x0, x1, x_t, t_expanded)
        
        # Forward pass
        v_pred = model(x_t, t)
        
        # Compute loss (square loss regression)
        loss = nn.MSELoss()(v_pred, v_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Save sample images every 100 batches
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # Save some generated samples
                sample_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Generate samples at t=0.5
                t_sample = torch.tensor([0.5], device=device).view(-1, 1, 1, 1)
                x_t_sample = interpolate_sample(x0[:8], x1[:8], t_sample)
                v_pred_sample = model(x_t_sample, torch.tensor([0.5], device=device).repeat(8))
                
                # Reconstruct x1 from x0 and predicted velocity
                x1_pred = x0[:8] + v_pred_sample
                x1_pred = torch.clamp(x1_pred, -1, 1)
                
                # Save images
                vutils.save_image(x1[:8], os.path.join(sample_dir, f'batch_{batch_idx:04d}_target.png'), 
                                normalize=True, nrow=4)
                vutils.save_image(x1_pred, os.path.join(sample_dir, f'batch_{batch_idx:04d}_pred.png'), 
                                normalize=True, nrow=4)
                
                if task == 'inpainting':
                    vutils.save_image(x0[:8], os.path.join(sample_dir, f'batch_{batch_idx:04d}_base.png'), 
                                    normalize=True, nrow=4)
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, task, save_dir, epoch):
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for x0, x1, mask in dataloader:
            x0, x1, mask = x0.to(device), x1.to(device), mask.to(device)
            batch_size = x0.size(0)
            
            # Sample time uniformly
            t = sample_time(batch_size, device)
            t_expanded = t.view(-1, 1, 1, 1)
            
            # Create interpolated sample
            x_t = interpolate_sample(x0, x1, t_expanded)
            
            # Compute target velocity
            v_target = compute_velocity(x0, x1, x_t, t_expanded)
            
            # Forward pass
            v_pred = model(x_t, t)
            
            # Compute loss
            loss = nn.MSELoss()(v_pred, v_target)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
    
    avg_loss = total_loss / num_samples
    
    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
    }, os.path.join(save_dir, f'model_epoch_{epoch:03d}.pth'))
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Stochastic Interpolants Reproduction')
    parser.add_argument('--task', type=str, default='inpainting', choices=['inpainting', 'super_resolution'],
                       help='Task to perform (inpainting or super_resolution)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    train_loader = get_dataloader(
        dataset_name='celebahq', 
        split='train', 
        image_size=args.image_size, 
        task=args.task, 
        batch_size=args.batch_size
    )
    
    val_loader = get_dataloader(
        dataset_name='celebahq', 
        split='test', 
        image_size=args.image_size, 
        task=args.task, 
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = StochasticInterpolant(in_channels=3, out_channels=3, base_channels=64).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"Starting training for {args.task} task...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.task, epoch, args.output)
        val_loss = evaluate_model(model, val_loader, device, args.task, args.output, epoch)
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print(f"Training completed. Results saved in {args.output}")

if __name__ == "__main__":
    main()