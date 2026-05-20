import os
import json
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import glob
from PIL import Image
import lpips
from pytorch_fid import fid_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cifar10_dataloaders(batch_size=64, seed=42, target_subset_size=10):
    transform = T.Compose([
        T.ToTensor(),
    ])

    # source: full CIFAR10 training set
    source_train = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # target: same dataset but we will take a random 10‑shot subset
    target_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = list(range(len(target_train)))
    random.shuffle(indices)
    target_indices = indices[:target_subset_size]
    target_subset = SubsetRandomSampler(target_indices)

    source_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    target_loader = DataLoader(target_train, batch_size=len(target_indices), sampler=target_subset, num_workers=4, pin_memory=True)

    return source_loader, target_loader, target_indices

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename, device='cpu'):
    return torch.load(filename, map_location=device)

def evaluate_fid(real_dir, gen_dir, batch_size=50, device='cuda'):
    return fid_score.calculate_fid_given_paths([real_dir, gen_dir], batch_size, device, 2048, 1)

def evaluate_lpips(real_dir, gen_dir):
    loss = lpips.LPIPS(net='vgg').cuda()
    real_imgs = []
    gen_imgs = []
    for f in sorted(glob.glob(os.path.join(real_dir, '*.png'))):
        img = Image.open(f).convert('RGB')
        real_imgs.append(np.array(img)/255.0)
    for f in sorted(glob.glob(os.path.join(gen_dir, '*.png'))):
        img = Image.open(f).convert('RGB')
        gen_imgs.append(np.array(img)/255.0)
    real_t = torch.tensor(real_imgs).permute(0,3,1,2).float().cuda()
    gen_t = torch.tensor(gen_imgs).permute(0,3,1,2).float().cuda()
    return loss(gen_t, real_t).mean().item()

def save_images(tensor_list, out_dir, prefix='img'):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(tensor_list):
        img = (img.detach().cpu() * 255).clamp(0,255).to(torch.uint8)
        img = img.permute(1,2,0).numpy()
        Image.fromarray(img).save(os.path.join(out_dir, f'{prefix}_{i:04d}.png'))

def collate_fn(batch):
    imgs, _ = zip(*batch)
    return torch.stack(imgs), torch.tensor([0]*len(imgs))