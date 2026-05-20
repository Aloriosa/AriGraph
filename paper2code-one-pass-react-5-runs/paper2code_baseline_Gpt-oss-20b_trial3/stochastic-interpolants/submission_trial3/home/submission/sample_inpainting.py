# sample_inpainting.py
import argparse
import os
import torch
import torchvision
from torchvision.utils import make_grid, save_image
from torchdiffeq import odeint
from tqdm import tqdm

from utils import create_random_mask, sample_base
from model import VelocityMLP

def ode_func(t, y, model, mask):
    """
    y: (B, C, H, W) – current state at time t
    t: scalar time in [0,1]
    """
    t = t.to(y.device)
    b = model(y, t)
    return b

def sample_one_image(model, device, img, mask, steps=1000):
    """
    Integrate the probability‑flow ODE from t=0 to t=1.
    """
    model.eval()
    with torch.no_grad():
        # Initial state: base sample
        x0 = sample_base(img, mask, noise_std=0.1).to(device)
        t_span = torch.linspace(0.0, 1.0, steps + 1, device=device)

        # odeint expects shape (steps, B, C, H, W) or (B, C, H, W)
        # We'll integrate batch of size 1
        sol = odeint(
            lambda t, y: ode_func(t, y, model, mask),
            x0,
            t_span,
            method='dopri5',
            rtol=1e-5,
            atol=1e-5,
        )
        # sol shape: (steps+1, B, C, H, W)
        xT = sol[-1]
        return xT.squeeze(0)  # (C, H, W)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = VelocityMLP().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Load MNIST test set
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output, exist_ok=True)

    for idx, (img, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(device)  # (1,1,28,28)
        # Random mask for this image
        mask = create_random_mask((1, 28, 28)).to(device)
        mask = mask.repeat(1, 1, 1, 1)

        # Generate sample
        sample = sample_one_image(model, device, img, mask, steps=args.steps)

        # Save
        out_path = os.path.join(args.output, f"sample_{idx:04d}.png")
        save_image(sample, out_path, normalize=True, range=(0, 1))
        if idx >= 999:  # save only first 1000 samples
            break

    print(f"Saved {idx+1} samples to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--output", type=str, default="samples", help="Directory to store samples")
    parser.add_argument("--steps", type=int, default=1000, help="ODE integration steps")
    args = parser.parse_args()
    main(args)