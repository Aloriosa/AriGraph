import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from datasets.cifar10 import CIFAR10Inpaint, CIFAR10SuperRes
from models.unet import UNetVelocity
from utils import get_device, ensure_dir, to_pil

def euler_integration(model, x0, t_steps, mask=None, device='cpu'):
    x = x0.clone()
    for t in t_steps:
        t_tensor = torch.full((x.shape[0],), t, device=device)
        b = model(x, t_tensor, mask=mask)
        x = x + b / len(t_steps)  # forward Euler, dt=1/len(t_steps)
    return x

def sample_inpainting(device, steps=10, batch_size=64, num_samples=5000):
    dataset = CIFAR10Inpaint(root='data', train=False, transform=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetVelocity(in_channels=3, base_channels=64).to(device)
    model.load_state_dict(torch.load('outputs/checkpoints/velocity_inpaint.pth', map_location=device))
    model.eval()

    ensure_dir('outputs/generated/inpainting')
    generated = []

    t_steps = torch.linspace(0, 1, steps=steps)

    with torch.no_grad():
        for x0, _, mask in tqdm(loader, desc='Sampling inpainting'):
            x0 = x0.to(device)
            mask = mask.to(device)
            # start at t=0
            x = euler_integration(model, x0, t_steps, mask=mask, device=device)
            generated.append(x.cpu())

            if len(generated) * batch_size >= num_samples:
                break

    all_gen = torch.cat(generated, dim=0)[:num_samples]
    ensure_dir('outputs/generated/inpainting')
    for i, img in enumerate(all_gen):
        to_pil(img).save(f'outputs/generated/inpainting/{i:05d}.png')
    print(f'{num_samples} inpainting samples saved.')

def sample_superres(device, steps=10, batch_size=64, num_samples=5000):
    dataset = CIFAR10SuperRes(root='data', train=False, transform=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetVelocity(in_channels=3, base_channels=64).to(device)
    model.load_state_dict(torch.load('outputs/checkpoints/velocity_sr.pth', map_location=device))
    model.eval()

    ensure_dir('outputs/generated/sr')
    generated = []

    t_steps = torch.linspace(0, 1, steps=steps)

    with torch.no_grad():
        for low, high in tqdm(loader, desc='Sampling superres'):
            low = low.to(device)
            # start at t=0
            x = euler_integration(model, low, t_steps, mask=None, device=device)
            generated.append(x.cpu())

            if len(generated) * batch_size >= num_samples:
                break

    all_gen = torch.cat(generated, dim=0)[:num_samples]
    for i, img in enumerate(all_gen):
        to_pil(img).save(f'outputs/generated/sr/{i:05d}.png')
    print(f'{num_samples} superresolution samples saved.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['inpainting', 'superresolution'], required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=5000)
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    if args.task == 'inpainting':
        sample_inpainting(device, steps=args.steps, num_samples=args.num_samples)
    else:
        sample_superres(device, steps=args.steps, num_samples=args.num_samples)

if __name__ == '__main__':
    main()