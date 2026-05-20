import os
import argparse
import torch
from torchvision import datasets, transforms
from torch_fidelity import calculate_metrics

def compute_fid(task, device):
    if task == 'inpainting':
        gen_dir = 'outputs/generated/inpainting'
        real_dir = 'data/CIFAR10/test'  # torchvision will not create this dir; we need to copy test set
    else:
        gen_dir = 'outputs/generated/sr'
        real_dir = 'data/CIFAR10/test'

    # Prepare real directory
    os.makedirs('data/CIFAR10/test', exist_ok=True)
    # Download test set images into directory
    test_set = datasets.CIFAR10(root='data', train=False, download=True)
    for idx, (img, _) in enumerate(test_set):
        img.save(f'data/CIFAR10/test/{idx:05d}.png')

    metrics = calculate_metrics(
        input1=gen_dir,
        input2=real_dir,
        cuda=device.type == 'cuda',
        net='inception',
        dims=2048,
        isc=False,
        fid=True,
        kid=False,
        verbose=True
    )
    return metrics['fid']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['inpainting', 'superresolution'], required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid = compute_fid(args.task, device)
    print(f'FID ({args.task}): {fid:.2f}')

if __name__ == '__main__':
    main()