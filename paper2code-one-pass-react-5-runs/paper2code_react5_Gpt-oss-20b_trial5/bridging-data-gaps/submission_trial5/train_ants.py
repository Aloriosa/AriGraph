"""
Minimal implementation of the DPMs‑ANT training loop.

We use a pre‑trained DDPM UNet from the `diffusers` library, a small
binary classifier trained on noisy images, and a finite‑step
gradient‑ascent routine for adversarial noise.

The script performs the following stages:
1. Train a binary classifier on noisy source (CIFAR‑10 train) and
   target (10‑shot cat) images.
2. Fine‑tune the UNet on the target domain using the similarity‑guided
   loss and adversarial noise selection.
3. Generate 1 000 samples with the fine‑tuned UNet.
4. Compute LPIPS & FID against the target set.
"""
import os
from pathlib import Path
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils import randn_tensor

import lpips
from data.target import load_target_dataset, download_and_prepare
from evaluate import compute_metrics

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS_CLF = 5
DEFAULT_EPOCHS_UNET = 3
DEFAULT_LR_CLF = 1e-3
DEFAULT_LR_UNET = 5e-5
DEFAULT_GAMMA = 5.0
DEFAULT_GEN_BATCH = 10
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_TRAIN_STEPS = 300  # approx 300 iterations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Simple classifier
# --------------------------------------------------------------------------- #
class SimpleClassifier(nn.Module):
    """
    Very small CNN that takes a noisy image and outputs logits for
    source(0)/target(1).
    """
    def __init__(self, img_shape=(3, 32, 32)):
        super().__init__()
        C, H, W = img_shape
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (H // 4) * (W // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def adversarial_noise_selection(unet, x0, t, eps_init, lr=0.02, steps=10):
    """
    Finite‑step gradient ascent to find a “worst‑case” noise for the
    current UNet at timestep t.  Returns the perturbed noise.
    """
    eps = eps_init.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([eps], lr=lr)
    alphas_cumprod = unet.scheduler.alphas_cumprod
    for _ in range(steps):
        optimizer.zero_grad()
        sqrt_at = torch.sqrt(torch.tensor(alphas_cumprod[t], device=eps.device))
        sqrt_1_at = torch.sqrt(1 - torch.tensor(alphas_cumprod[t], device=eps.device))
        xt = sqrt_at * x0 + sqrt_1_at * eps
        eps_pred = unet(xt, t, return_dict=False)[0]
        loss = torch.mean((eps - eps_pred) ** 2)
        loss.backward()
        optimizer.step()
        # Re‑norm the noise to have mean 0 and std 1
        eps.data = eps.data - eps.data.mean()
        eps.data = eps.data / (eps.data.std() + 1e-6)
    return eps.detach()


def train_classifier(classifier, dataloader, scheduler, epochs, lr, device):
    """Train the binary classifier on noisy images."""
    classifier.to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    alphas_cumprod = scheduler.alphas_cumprod

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Clf epoch {epoch+1}/{epochs}"):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.shape[0]

            # Sample random timesteps
            t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
            eps = torch.randn_like(imgs, device=device)

            # Noised images
            sqrt_at = torch.sqrt(torch.tensor(alphas_cumprod[t], device=device))
            sqrt_1_at = torch.sqrt(1 - torch.tensor(alphas_cumprod[t], device=device))
            xt = sqrt_at * imgs + sqrt_1_at * eps

            logits = classifier(xt)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Classifier epoch {epoch+1} loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(classifier.state_dict(), "checkpoints/classifier.pt")
    return classifier


def train_unet(unet, classifier, scheduler, target_loader, epochs, lr, gamma, device):
    """Fine‑tune the UNet on the target domain."""
    unet.to(device)
    classifier.eval()  # keep classifier fixed
    optimizer = optim.AdamW(unet.parameters(), lr=lr)
    alphas_cumprod = scheduler.alphas_cumprod

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in tqdm(target_loader, desc=f"UNet epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]

            # Random timesteps
            t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
            eps = torch.randn_like(imgs, device=device)

            # Adversarial noise selection
            eps_star = adversarial_noise_selection(unet, imgs, t, eps)

            # Noised image with worst‑case noise
            sqrt_at = torch.sqrt(torch.tensor(alphas_cumprod[t], device=device))
            sqrt_1_at = torch.sqrt(1 - torch.tensor(alphas_cumprod[t], device=device))
            xt = sqrt_at * imgs + sqrt_1_at * eps_star

            # Similarity‑guided term
            logits = classifier(xt)
            probs = torch.softmax(logits, dim=-1)
            target_prob = probs[:, 1]  # probability of target class
            log_prob = torch.log_softmax(logits, dim=-1)[:, 1]
            # Gradient of log p(y=target | x_t)
            grad_log_p = torch.autograd.grad(
                log_prob.sum(), xt, create_graph=True
            )[0]

            eps_pred = unet(xt, t, return_dict=False)[0]
            loss = torch.mean((eps_star - eps_pred - gamma * grad_log_p) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"UNet epoch {epoch+1} loss: {epoch_loss / len(target_loader):.4f}")

    torch.save(unet.state_dict(), "checkpoints/unet.pt")
    return unet


def generate_samples(unet, scheduler, num_samples, batch_size, device):
    """Generate images with the fine‑tuned UNet."""
    os.makedirs("generated", exist_ok=True)
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)

    all_imgs = []
    with torch.no_grad():
        for _ in tqdm(
            range(0, num_samples, batch_size),
            desc="Generating samples",
        ):
            num = min(batch_size, num_samples - len(all_imgs))
            samples = pipeline(
                num_inference_steps=scheduler.num_train_timesteps,
                generator=None,
                output_type="pt",
                batch_size=num,
            ).images
            all_imgs.append(samples.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)
    # Convert to [0,1] and save
    for i, img in enumerate(all_imgs):
        img = img.clamp(0, 1)
        torchvision.utils.save_image(
            img, f"generated/img_{i:04d}.png", normalize=False, nrow=1
        )
    return "generated"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    # 1. Prepare data
    print("Downloading & preparing target dataset...")
    download_and_prepare(num_samples=10, seed=42)
    target_loader = load_target_dataset(batch_size=args.batch_size)

    # Source dataset (CIFAR‑10 train) for classifier training
    transform = transforms.Compose([transforms.ToTensor()])
    source_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    source_loader = DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Combine source (label 0) & target (label 1) for classifier
    class_labels_source = torch.zeros(len(source_dataset), dtype=torch.long)
    class_labels_target = torch.ones(len(target_loader.dataset), dtype=torch.long)
    combined_dataset = ConcatDataset(
        [
            source_dataset,
            target_loader.dataset,
        ]
    )
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
    )

    # 2. Load pre‑trained DDPM UNet & scheduler
    print("Loading pre‑trained DDPM model...")
    unet = UNet2DModel.from_pretrained(
        "google/ddpm-celebahq-256",
        subfolder="unet",
    ).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(
        "google/ddpm-celebahq-256",
        subfolder="scheduler",
    )

    # 3. Train classifier
    print("Training binary classifier...")
    classifier = SimpleClassifier(img_shape=(3, 32, 32))
    classifier = train_classifier(
        classifier,
        combined_loader,
        scheduler,
        epochs=args.clf_epochs,
        lr=args.clf_lr,
        device=DEVICE,
    )

    # 4. Fine‑tune UNet
    print("Fine‑tuning UNet on target domain...")
    unet = train_unet(
        unet,
        classifier,
        scheduler,
        target_loader,
        epochs=args.unet_epochs,
        lr=args.unet_lr,
        gamma=args.gamma,
        device=DEVICE,
    )

    # 5. Generate samples
    print("Generating samples...")
    gen_dir = generate_samples(
        unet, scheduler, args.num_samples, args.gen_batch, DEVICE
    )

    # 6. Evaluate
    print("Evaluating metrics...")
    compute_metrics(gen_dir, "data/target_images", args)
    print("Reproduction completed. Metrics are stored in metrics.txt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DPMs‑ANT training")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clf_epochs", type=int, default=DEFAULT_EPOCHS_CLF)
    parser.add_argument("--clf_lr", type=float, default=DEFAULT_LR_CLF)
    parser.add_argument("--unet_epochs", type=int, default=DEFAULT_EPOCHS_UNET)
    parser.add_argument("--unet_lr", type=float, default=DEFAULT_LR_UNET)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gen_batch", type=int, default=DEFAULT_GEN_BATCH)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    args = parser.parse_args()
    main(args)