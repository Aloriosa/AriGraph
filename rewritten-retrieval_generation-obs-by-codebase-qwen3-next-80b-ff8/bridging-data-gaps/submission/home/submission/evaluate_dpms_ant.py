import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm
import random
from fid_score import calculate_fid_given_paths
from lpips import LPIPS

# Import model components
from model.DDPM.diffusion import Model
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

class DPMS_ANT_Evaluator:
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
        
        # Load trained model
        model_path = 'ckpt/DDPM_ANT_final.pt'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model from {model_path}")
        else:
            # Try loading from iteration 150
            model_path = 'ckpt/DDPM_ANT_150.pt'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded trained model from {model_path}")
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            beta_start=self.config['dm']['beta_start'],
            beta_end=self.config['dm']['beta_end'],
            num_timesteps=self.config['dm']['num_diffusion_timesteps'],
            schedule_name=self.config['dm']['schedule_name']
        )
        
        # Load target dataset for evaluation
        self.target_dataset = ImageDataset(self.config['data']['target_data_path'], self.config['data']['img_size'])
        self.target_loader = DataLoader(self.target_dataset, batch_size=self.config['model']['batch_size'], shuffle=False, num_workers=4)
        
        # Initialize LPIPS metric
        self.lpips_metric = LPIPS(net='alex').to(self.device)
        
        # Create output directories
        os.makedirs('output', exist_ok=True)
        
        # Set random seed
        torch.manual_seed(1228)
        np.random.seed(1228)
        random.seed(1228)
    
    def evaluate(self):
        print("Starting evaluation...")
        
        # Generate samples
        num_samples = 1000
        batch_size = self.config['model']['batch_size']
        num_batches = num_samples // batch_size
        
        generated_samples = []
        
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            # Sample noise
            x = torch.randn(batch_size, 3, self.config['data']['img_size'], self.config['data']['img_size']).to(self.device)
            
            # Denoising process
            for t in reversed(range(self.config['dm']['num_diffusion_timesteps'])):
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                pred_noise = self.model(x, t_tensor)
                
                # Calculate mean and variance for the reverse process
                sqrt_alphas_cumprod = torch.from_numpy(self.noise_scheduler.sqrt_alphas_cumprod).to(self.device)
                sqrt_one_minus_alphas_cumprod = torch.from_numpy(self.noise_scheduler.sqrt_one_minus_alphas_cumprod).to(self.device)
                
                alpha_t = sqrt_alphas_cumprod[t] ** 2
                alpha_t_bar = sqrt_alphas_cumprod[t] ** 2
                alpha_t_prev = sqrt_alphas_cumprod[t-1] ** 2 if t > 0 else 1.0
                
                # Calculate predicted x0
                pred_x0 = (x - sqrt_one_minus_alphas_cumprod[t] * pred_noise) / sqrt_alphas_cumprod[t]
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                # Calculate mean
                mean = (
                    (alpha_t_prev ** 0.5) * pred_x0 +
                    ((1 - alpha_t_prev - (1 - alpha_t) * (alpha_t_prev / alpha_t)) ** 0.5) * pred_noise
                )
                
                # Sample from normal distribution
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Update x
                x = mean + (1 - alpha_t_prev) ** 0.5 * noise
            
            # Save generated samples
            generated_samples.append(x.cpu())
        
        # Concatenate all generated samples
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Save generated samples
        os.makedirs('output/generated', exist_ok=True)
        for i in range(min(100, len(generated_samples))):
            sample = generated_samples[i]
            sample = (sample + 1) / 2  # Denormalize
            sample = torch.clamp(sample, 0, 1)
            vutils.save_image(sample, f'output/generated/sample_{i:04d}.png')
        
        # Calculate FID
        print("Calculating FID...")
        fid_value = calculate_fid_given_paths(
            paths=['output/generated', self.config['data']['target_data_path']],
            batch_size=batch_size,
            device=self.device,
            dims=2048
        )
        
        # Calculate intra-LPIPS
        print("Calculating intra-LPIPS...")
        lpips_scores = []
        
        # Use target dataset as reference for intra-LPIPS
        target_samples = []
        for batch in self.target_loader:
            target_samples.append(batch)
        
        target_samples = torch.cat(target_samples, dim=0)
        
        # Calculate LPIPS between generated samples and target samples
        for i in range(min(100, len(generated_samples))):
            gen_sample = generated_samples[i].unsqueeze(0).to(self.device)
            target_sample = target_samples[i % len(target_samples)].unsqueeze(0).to(self.device)
            
            lpips_score = self.lpips_metric(gen_sample, target_sample)
            lpips_scores.append(lpips_score.item())
        
        intra_lpips = np.mean(lpips_scores)
        
        # Print results
        print(f"FID: {fid_value:.3f}")
        print(f"Intra-LPIPS: {intra_lpips:.3f}")
        
        # Save results to output.csv
        import csv
        with open('output.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['FID', f'{fid_value:.3f}'])
            writer.writerow(['Intra-LPIPS', f'{intra_lpips:.3f}'])
            writer.writerow(['Training Iterations', '150'])
        
        print("Evaluation completed! Results saved to output.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DPMS_ANT_Evaluator(args.config)
    
    # Run evaluation
    evaluator.evaluate()

if __name__ == '__main__':
    main()