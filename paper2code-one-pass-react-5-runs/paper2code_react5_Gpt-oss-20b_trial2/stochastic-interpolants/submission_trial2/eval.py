import argparse
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torchmetrics.image.fid import FID


def load_real(num_samples: int) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_samples, shuffle=False, num_workers=4, drop_last=False
    )
    real = next(iter(loader))[0]  # (B,3,H,W)
    return real


def load_gen(gen_dir: str, num_samples: int) -> torch.Tensor:
    imgs = []
    for fname in sorted(os.listdir(gen_dir)):
        if fname.endswith(".png"):
            img = torchvision.io.read_image(os.path.join(gen_dir, fname)).float() / 255.0
            imgs.append(img)
            if len(imgs) >= num_samples:
                break
    gen = torch.stack(imgs)  # (B,3,H,W)
    return gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="fid.txt")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    num_samples = args.num_samples
    real = load_real(num_samples).to("cpu")
    gen = load_gen(args.gen_dir, num_samples).to("cpu")

    fid = FID().to(real.device)
    fid_score = fid(real, gen)
    with open(args.output, "w") as f:
        f.write(f"FID: {fid_score.item():.4f}\n")
    print(f"FID: {fid_score.item():.4f}")


if __name__ == "__main__":
    main()