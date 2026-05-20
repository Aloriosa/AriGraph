import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class ImageDataset(Dataset):
    def __init__(self, dataset_name='celebahq', split='train', image_size=64, 
                 task='inpainting', mask_type='center', super_resolution_factor=4):
        self.image_size = image_size
        self.task = task
        self.mask_type = mask_type
        self.super_resolution_factor = super_resolution_factor
        
        # Load dataset
        if dataset_name == 'celebahq':
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.dataset = torchvision.datasets.CelebA(
                root='./data', 
                split='train' if split == 'train' else 'test',
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        self.length = len(self.dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        
        if self.task == 'inpainting':
            # Create mask (center square)
            mask = torch.ones_like(img)
            mask_size = self.image_size // 4
            start = (self.image_size - mask_size) // 2
            mask[:, start:start+mask_size, start:start+mask_size] = 0
            
            # Create base sample: x0 = x1 ⊕ (1 - mask) ⊙ ε
            noise = torch.randn_like(img)
            base_sample = img * mask + noise * (1 - mask)
            
            return base_sample, img, mask
            
        elif self.task == 'super_resolution':
            # Create low-resolution version
            hr_img = img
            lr_img = F.interpolate(img.unsqueeze(0), 
                                  scale_factor=1/self.super_resolution_factor, 
                                  mode='bilinear', 
                                  align_corners=False).squeeze(0)
            lr_img = F.interpolate(lr_img.unsqueeze(0), 
                                  size=(self.image_size, self.image_size), 
                                  mode='bilinear', 
                                  align_corners=False).squeeze(0)
            
            return lr_img, hr_img, torch.ones_like(img)  # No mask for super-resolution
            
        else:
            raise ValueError(f"Task {self.task} not supported")

def get_dataloader(dataset_name='celebahq', split='train', image_size=64, 
                   task='inpainting', batch_size=32, num_workers=4):
    dataset = ImageDataset(dataset_name, split, image_size, task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=num_workers, pin_memory=True)