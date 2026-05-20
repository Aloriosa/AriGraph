import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import os

class InceptionV3(nn.Module):
    def __init__(self, resize_input=True, normalize_input=True):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval()
    
    def forward(self, x):
        if self.resize_input:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = (x - 0.5) / 0.5
        
        return self.model(x)

def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cuda'):
    model.eval()
    
    if len(images) == 0:
        return np.zeros(dims), np.zeros((dims, dims))
    
    pred = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            pred.append(model(batch.to(device)).cpu().numpy())
    
    pred = np.concatenate(pred, axis=0)
    
    mu = np.mean(pred, axis=0)
    sigma = np.cov(pred, rowvar=False)
    
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %d to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid_given_paths(paths, batch_size=50, device='cuda', dims=2048):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.default_block_idx
    
    model = InceptionV3().to(device)
    
    # Load images from paths
    def load_images_from_path(path):
        images = []
        for filename in os.listdir(path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(path, filename)).convert('RGB')
                img = transforms.ToTensor()(img)
                images.append(img)
        return torch.stack(images)
    
    images1 = load_images_from_path(paths[0])
    images2 = load_images_from_path(paths[1])
    
    # Calculate statistics
    m1, s1 = calculate_activation_statistics(images1, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(images2, model, batch_size, dims, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value