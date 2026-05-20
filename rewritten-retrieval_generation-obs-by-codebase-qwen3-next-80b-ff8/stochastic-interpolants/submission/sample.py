import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import torchvision.utils as vutils

from models import UNet
from data import ImageInpaintingDataset, ImageSuperResolutionDataset
from utils.noise import NoiseSchedule
from utils.sde import SDESolver

def sample_images(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    if args.task == 'inpainting':
        model = UNet(in_channels=3, out_channels=3).to(device)
    else:  # super_resolution
        model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize noise schedule
    noise_schedule = NoiseSchedule(timesteps=1000, schedule='linear')
    
    # Initialize SDE solver
    solver = SDESolver(model, noise_schedule, timesteps=1000)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate samples
    num_samples = 16
    if args.task == 'inpainting':
        # Create a simple mask for sampling
        mask = torch.zeros(1, 3, 64, 64)
        mask[:, :, 16:48, 16:48] = 1  # Center mask
        
        # Generate samples
        with torch.no_grad():
            samples = []
            for i in range(num_samples):
                sample = solver.sample_with_mask(
                    (1, 3, 64, 64), 
                    device, 
                    mask.to(device), 
                    verbose=True
                )
                samples.append(sample)
            
            samples = torch.cat(samples, dim=0)
            vutils.save_image(
                samples, 
                os.path.join(args.output, 'samples_inpainting.png'), 
                nrow=4, 
                normalize=True, 
                range=(-1, 1)
            )
            
    else:  # super_resolution
        # Generate low-res images
        with torch.no_grad():
            samples = []
            for i in range(num_samples):
                # Start with random noise
                low_res = torch.randn(1, 3, 64, 64, device=device)
                # Upsample to target size
                high_res = solver.sample(
                    (1, 3, 256, 256), 
                    device, 
                    verbose=True
                )
                samples.append(high_res)
            
            samples = torch.cat(samples, dim=0)
            vutils.save_image(
                samples, 
                os.path.join(args.output, 'samples_super_resolution.png'), 
                nrow=4, 
                normalize=True, 
                range=(-1, 1)
            )
    
    print(f'Samples saved to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['inpainting', 'super_resolution'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='samples')
    
    args = parser.parse_args()
    sample_images(args)