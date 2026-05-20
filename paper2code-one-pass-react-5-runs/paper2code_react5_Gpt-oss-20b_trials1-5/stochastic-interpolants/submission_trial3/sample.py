import argparse
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from utils import random_mask, get_time_embedding
from models import UNetSmall

def get_dataloader(batch_size, split='train'):
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=(split=='train'),
                                          download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=2, pin_memory=True)
    return loader

def integrate(model, x0, mask, steps=1000, device='cpu'):
    """Forward Euler integration of the probability‑flow ODE."""
    dt = 1.0 / steps
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0], 1, 1, 1), i / steps, device=device)
        t_emb = get_time_embedding(t.squeeze(3).squeeze(2))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
        inp = torch.cat([x, mask, t_emb], dim=1)  # Bx5xHxW
        v = model(inp)  # Bx3xHxW
        x = x + dt * v
    return x

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    loader = get_dataloader(args.batch_size, split='train')

    samples_done = 0
    for imgs, _ in tqdm(loader, desc='Sampling'):
        if samples_done >= args.num_samples:
            break
        imgs = imgs.to(device)
        B = imgs.shape[0]
        # Random mask for each image
        mask = random_mask(B, 32, 32, p=0.3).to(device)

        # Base sample x0 = mask * imgs + (1-mask) * noise
        noise = torch.randn_like(imgs)
        x0 = mask * imgs + (1 - mask) * noise

        # Integrate
        with torch.no_grad():
            sample = integrate(model, x0, mask, steps=args.steps, device=device)

        # Save images
        for i in range(B):
            if samples_done >= args.num_samples:
                break
            img = sample[i].cpu()
            torchvision.utils.save_image(img, os.path.join(args.output,
                                                          f'sample_{samples_done:04d}.png'))
            samples_done += 1

    print(f'Saved {samples_done} samples to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for sampling')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of ODE integration steps')
    parser.add_argument('--output', type=str, default='samples',
                        help='Output directory for generated images')
    args = parser.parse_args()
    main(args)