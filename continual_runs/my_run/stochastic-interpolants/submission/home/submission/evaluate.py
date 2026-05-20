import torch
import numpy as np
import os
import argparse
import csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import torchvision.transforms as transforms

def load_image(path):
    """Load and preprocess image"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)

def compute_psnr_ssim(img1_path, img2_path):
    """Compute PSNR and SSIM between two images"""
    img1 = load_image(img1_path).numpy()
    img2 = load_image(img2_path).numpy()
    
    # Convert from [0,1] to [0,255]
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    
    # Per-channel PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    
    # SSIM (using all channels)
    ssim = structural_similarity(img1, img2, multichannel=True, data_range=255)
    
    return psnr, ssim

def evaluate_results(input_dir, output_file):
    """Evaluate results from training"""
    results = []
    
    # Find all prediction and target images
    for epoch_dir in sorted(os.listdir(input_dir)):
        if not epoch_dir.startswith('epoch_'):
            continue
            
        epoch_num = int(epoch_dir.split('_')[1])
        
        # Look for prediction and target images
        pred_files = [f for f in os.listdir(os.path.join(input_dir, epoch_dir)) if f.endswith('_pred.png')]
        target_files = [f for f in os.listdir(os.path.join(input_dir, epoch_dir)) if f.endswith('_target.png')]
        
        if len(pred_files) == 0:
            continue
            
        # Calculate average PSNR and SSIM for this epoch
        psnr_scores = []
        ssim_scores = []
        
        for pred_file in pred_files:
            # Match with corresponding target file
            target_file = pred_file.replace('_pred.png', '_target.png')
            pred_path = os.path.join(input_dir, epoch_dir, pred_file)
            target_path = os.path.join(input_dir, epoch_dir, target_file)
            
            if os.path.exists(target_path):
                psnr, ssim = compute_psnr_ssim(pred_path, target_path)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
        
        if len(psnr_scores) > 0:
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)
            results.append({
                'epoch': epoch_num,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'samples': len(psnr_scores)
            })
    
    # Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'psnr', 'ssim', 'samples'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Stochastic Interpolants Results')
    parser.add_argument('--input', type=str, required=True, help='Input directory with results')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    
    args = parser.parse_args()
    evaluate_results(args.input, args.output)

if __name__ == "__main__":
    main()