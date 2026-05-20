import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import torchvision.utils as vutils

from models import UNet
from data import ImageInpaintingDataset, ImageSuperResolutionDataset
from utils.noise import NoiseSchedule, sample_base_density, corrupt_image
from utils.sde import SDESolver

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize datasets
    if args.task == 'inpainting':
        dataset = ImageInpaintingDataset(
            root_dir=args.data_dir, 
            mask_type=args.mask_type, 
            mask_size=args.mask_size
        )
    elif args.task == 'super_resolution':
        dataset = ImageSuperResolutionDataset(
            root_dir=args.data_dir, 
            scale_factor=args.scale_factor
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    if args.task == 'inpainting':
        model = UNet(in_channels=3, out_channels=3).to(device)
    else:  # super_resolution
        model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Initialize noise schedule
    noise_schedule = NoiseSchedule(timesteps=args.timesteps, schedule='linear')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if args.task == 'inpainting':
                corrupted_image, target_image, mask = batch
                corrupted_image = corrupted_image.to(device)
                target_image = target_image.to(device)
                mask = mask.to(device)
                
                # Sample time steps
                t = torch.randint(0, args.timesteps, (corrupted_image.size(0),), device=device).float() / args.timesteps
                
                # Sample base density using data-dependent coupling
                # For training, we use the corrupted image as base and target image as target
                base_samples = corrupted_image
                target_samples = target_image
                
                # Predict velocity
                velocity_pred = model(base_samples, t)
                
                # Compute true velocity (approximate using difference)
                # In practice, this would be derived from the data-dependent coupling
                # For simplicity, we use the difference between target and base
                velocity_true = (target_samples - base_samples) / (1 - t.view(-1, 1, 1, 1) + 1e-8)
                
                # Compute loss
                loss = criterion(velocity_pred, velocity_true)
                
            else:  # super_resolution
                low_res_image, high_res_image = batch
                low_res_image = low_res_image.to(device)
                high_res_image = high_res_image.to(device)
                
                # Sample time steps
                t = torch.randint(0, args.timesteps, (low_res_image.size(0),), device=device).float() / args.timesteps
                
                # Use low-res as base, high-res as target
                base_samples = low_res_image
                target_samples = high_res_image
                
                # Predict velocity
                velocity_pred = model(base_samples, t)
                
                # Compute true velocity
                velocity_true = (target_samples - base_samples) / (1 - t.view(-1, 1, 1, 1) + 1e-8)
                
                # Compute loss
                loss = criterion(velocity_pred, velocity_true)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, 'checkpoint_best.pth'))
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f'Training completed. Best loss: {best_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='inpainting', choices=['inpainting', 'super_resolution'])
    parser.add_argument('--data_dir', type=str, default='/home/submission/data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='results/inpainting')
    parser.add_argument('--mask_type', type=str, default='center', choices=['center', 'random'])
    parser.add_argument('--mask_size', type=int, default=64)
    parser.add_argument('--scale_factor', type=int, default=4)
    
    args = parser.parse_args()
    
    train_model(args)