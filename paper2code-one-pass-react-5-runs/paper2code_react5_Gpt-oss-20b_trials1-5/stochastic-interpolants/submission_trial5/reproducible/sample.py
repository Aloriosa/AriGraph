import argparse
import yaml
import torch
import os
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm

from .model import create_velocity_model
from .utils import get_time_embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoints/ckpt_epoch_10.pt')
    return parser.parse_args()

def velocity_func(model, t, x):
    """
    ODE: dX/dt = v(t, X)
    """
    # t is a scalar tensor
    return model(x, t)

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'])
    model = create_velocity_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Sample base points x0 ~ N(0, sigma^2 I)
    sigma = cfg['sigma']
    num_samples = 16  # keep it small for quick demo
    x0 = sigma * torch.randn(num_samples, 3, 256, 256, device=device)

    # Time grid
    steps = 100
    t = torch.linspace(0., 1., steps, device=device)

    with torch.no_grad():
        # ODE integration
        samples = odeint(lambda t, y: velocity_func(model, t, y),
                         x0, t, method='dopri5')
        # samples shape: [steps, B, C, H, W]
        x1 = samples[-1]  # final time

    os.makedirs('samples', exist_ok=True)
    for i in range(num_samples):
        img = (x1[i].cpu() + 1) / 2  # back to [0,1]
        torchvision.utils.save_image(img, f'samples/sample_{i}.png')

    print(f"Saved {num_samples} samples to samples/")

if __name__ == "__main__":
    main()