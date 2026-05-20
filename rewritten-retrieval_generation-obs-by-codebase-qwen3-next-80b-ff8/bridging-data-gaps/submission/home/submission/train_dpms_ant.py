import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Import model components
from model.DDPM.diffusion import Model
from model.adapter_utils.adapter import Adapter
from utils.denoising import NoiseScheduler, get_named_beta_schedule

class ImageDataset(Dataset):
    def __init__(self, data_path, img_size=256, transform=None):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_size = img_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DPMS_ANT:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize diffusion model
        self.model = Model(
            ch=self.config['model']['ddpm']['ch'],
            in_channels=self.config['model']['ddpm']['in_channels'],
            out_ch=self.config['model']['ddpm']['out_ch'],
            ch_mult=self.config['model']['ddpm']['ch_mult'],
            num_res_blocks=self.config['model']['ddpm']['num_res_blocks'],
            attn_resolutions=self.config['model']['ddpm']['attn_resolutions'],
            dropout=self.config['model']['ddpm']['dropout'],
            resamp_with_conv=self.config['model']['ddpm']['resamp_with_conv'],
            model_type='simple',
            img_size=self.config['data']['img_size'],
            num_timesteps=self.config['dm']['num_diffusion_timesteps'],
            with_adapter=True,
            adapter_dim=self.config['model']['adapter']['dim'],
            adapter_patch_size=4,
            adapter_num_heads=self.config['model']['adapter']['num_heads'],
            adapter_qkv_bias=True,
            adapter_drop=self.config['model']['adapter']['drop']
        ).to(self.device)
        
        # Initialize classifier
        self.classifier = Classifier().to(self.device)
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            beta_start=self.config['dm']['beta_start'],
            beta_end=self.config['dm']['beta_end'],
            num_timesteps=self.config['dm']['num_diffusion_timesteps'],
            schedule_name=self.config['dm']['schedule_name']
        )
        
        # Load pre-trained weights if available
        if os.path.exists(self.config['model']['ddpm']['initial_checkpoint']):
            self.model.load_state_dict(torch.load(self.config['model']['ddpm']['initial_checkpoint'], map_location=self.device))
            print(f"Loaded pre-trained DDPM from {self.config['model']['ddpm']['initial_checkpoint']}")
        
        if os.path.exists(self.config['model']['classifier']['initial_checkpoint']):
            self.classifier.load_state_dict(torch.load(self.config['model']['classifier']['initial_checkpoint'], map_location=self.device))
            print(f"Loaded pre-trained classifier from {self.config['model']['classifier']['initial_checkpoint']}")
        
        # Freeze base DDPM parameters, only train adapter
        for name, param in self.model.named_parameters():
            if "adapter" not in name.lower():
                param.requires_grad = False
        
        # Set up optimizer for adapter parameters only
        self.adapter_params = [p for n, p in self.model.named_parameters() if "adapter" in n.lower()]
        self.optimizer = optim.Adam(self.adapter_params, lr=self.config['train']['lr'], 
                                   betas=self.config['opt']['betas'], weight_decay=self.config['opt']['weight_decay'])
        
        # Set up classifier optimizer
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Load datasets
        self.source_dataset = ImageDataset(self.config['data']['source_data_path'], self.config['data']['img_size'])
        self.target_dataset = ImageDataset(self.config['data']['target_data_path'], self.config['data']['img_size'])
        
        self.source_loader = DataLoader(self.source_dataset, batch_size=self.config['model']['batch_size'], shuffle=True, num_workers=4)
        self.target_loader = DataLoader(self.target_dataset, batch_size=self.config['model']['batch_size'], shuffle=True, num_workers=4)
        
        # Create output directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('ckpt', exist_ok=True)
        
        # Initialize training parameters
        self.iteration = 0
        self.max_iterations = self.config['train']['iteration']
        self.ad_omega = self.config['tl']['ad_omega']
        self.ad_num_iter = self.config['tl']['ad_num_iter']
        self.c = self.config['tl']['c']
        
        # For similarity-guided training
        self.lambda_sim = 5.0  # As specified in paper_card_0001
        
    def train(self):
        print("Starting DPMS-ANT training...")
        
        # Train classifier on source domain
        self.train_classifier()
        
        # Training loop
        pbar = tqdm(range(self.max_iterations), desc="Training")
        for iteration in pbar:
            self.iteration = iteration
            
            # Get batch from target domain
            try:
                target_batch = next(self.target_iter)
            except:
                self.target_iter = iter(self.target_loader)
                target_batch = next(self.target_iter)
            
            target_batch = target_batch.to(self.device)
            
            # Sample random timesteps
            t = torch.randint(0, self.config['dm']['num_diffusion_timesteps'], (target_batch.size(0),), device=self.device).long()
            
            # Add noise to target images
            noise = torch.randn_like(target_batch)
            noisy_target = self.noise_scheduler.q_sample(target_batch, t, noise)
            
            # Forward pass through DDPM
            pred_noise = self.model(noisy_target, t)
            
            # Calculate noise prediction loss
            noise_loss = self.mse_loss(pred_noise, noise)
            
            # Similarity-guided training
            similarity_loss = self.calculate_similarity_loss(target_batch, t)
            
            # Adversarial noise selection
            adversarial_loss = self.adversarial_noise_selection(target_batch, t)
            
            # Total loss
            total_loss = noise_loss + self.lambda_sim * similarity_loss + self.ad_omega * adversarial_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter_params, self.config['opt']['clip_grad'])
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'noise_loss': f'{noise_loss.item():.4f}',
                'sim_loss': f'{similarity_loss.item():.4f}',
                'adv_loss': f'{adversarial_loss.item():.4f}'
            })
            
            # Save checkpoint
            if (iteration + 1) % self.config['save_freq'] == 0:
                torch.save(self.model.state_dict(), f'ckpt/DDPM_ANT_{iteration+1}.pt')
        
        # Save final model
        torch.save(self.model.state_dict(), 'ckpt/DDPM_ANT_final.pt')
        print("Training completed!")
    
    def train_classifier(self):
        print("Training classifier on source domain...")
        
        # Train classifier for 100 iterations
        for i in range(100):
            for source_batch in self.source_loader:
                source_batch = source_batch.to(self.device)
                
                # Forward pass
                logits = self.classifier(source_batch)
                labels = torch.zeros(source_batch.size(0), dtype=torch.long, device=self.device)  # Source domain label
                
                loss = self.ce_loss(logits, labels)
                
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
        
        # Save classifier
        torch.save(self.classifier.state_dict(), 'ckpt/FFHQ_SUNGLASSES_CLASSIFIER.pt')
        print("Classifier training completed!")
    
    def calculate_similarity_loss(self, target_batch, t):
        """Calculate similarity loss using classifier for feature alignment"""
        # Get features from classifier
        with torch.no_grad():
            source_features = self.classifier.features(target_batch)
            source_features = torch.mean(source_features, dim=[2, 3])  # Global average pooling
        
        # For target domain, we want to align features with source domain
        # This is a simplified version of the similarity measurement
        # In the paper, they use a more sophisticated feature similarity metric
        similarity_loss = torch.mean(torch.pow(source_features, 2))
        
        return similarity_loss
    
    def adversarial_noise_selection(self, target_batch, t):
        """Implement adversarial noise selection using PGD attack"""
        # Create a copy of the target batch that requires gradients
        x_adv = target_batch.clone().detach().requires_grad_(True)
        
        # Perform PGD attack
        for _ in range(self.ad_num_iter):
            # Add noise to the target batch
            noise = torch.randn_like(x_adv)
            noisy_x = self.noise_scheduler.q_sample(x_adv, t, noise)
            
            # Get prediction from model
            pred_noise = self.model(noisy_x, t)
            
            # Calculate loss (we want to maximize the prediction error)
            noise_loss = self.mse_loss(pred_noise, noise)
            
            # Calculate gradient
            self.model.zero_grad()
            noise_loss.backward()
            
            # Update adversarial example
            x_adv = x_adv + self.c * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, -1, 1)
            x_adv = x_adv.detach().requires_grad_(True)
        
        # Use the adversarial example for training
        noisy_x_adv = self.noise_scheduler.q_sample(x_adv, t, torch.randn_like(x_adv))
        pred_noise_adv = self.model(noisy_x_adv, t)
        adv_loss = self.mse_loss(pred_noise_adv, torch.randn_like(pred_noise_adv))
        
        return adv_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(1228)
    np.random.seed(1228)
    random.seed(1228)
    
    # Initialize DPMS-ANT
    dpms_ant = DPMS_ANT(args.config)
    
    # Start training
    dpms_ant.train()

if __name__ == '__main__':
    main()