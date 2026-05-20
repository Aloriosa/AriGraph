#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchattacks
import clip
from torchvision import datasets, transforms
from utils import load_clip_model

def parse_args():
    parser = argparse.ArgumentParser(description="FARE‑CLIP training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet validation subset (used as training data).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=str, default="4/255",
                        help="Adversarial epsilon, e.g. '4/255'")
    parser.add_argument("--adv_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints/fare_clip")
    return parser.parse_args()

def run(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load a fresh CLIP (original) for reference embeddings
    clip_org, preprocess = load_clip_model(device)
    clip_org.eval()

    # Load a copy of the vision encoder to fine‑tune
    visual = clip_org.visual
    # We keep the text encoder frozen
    text_encoder = clip_org.encode_text

    # Wrap the visual encoder as a trainable module
    model = nn.Sequential(visual)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # Loss weight for the adversarial term
    adv_weight = 1.0

    # Attack object (APGD) that maximizes the embedding distance
    eps_val = eval(args.epsilon)  # e.g. 4/255
    attack = torchattacks.APGD(
        model,
        loss_func=lambda y, y_hat: -((y_hat - y).pow(2).sum(-1)).mean(),  # maximize L2 distance
        eps=eps_val,
        alpha=eps_val / args.adv_steps,
        steps=args.adv_steps,
        rand_init=True,
        rand_init_eps=eps_val,
    )

    # Data loader (ImageNet validation subset as training data)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_org.visual.input_mean, std=clip_org.visual.input_std),
    ])
    train_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
        for images, _ in train_loader:
            images = images.to(device)

            # Original embeddings (reference)
            with torch.no_grad():
                phi_org = visual(images)

            # 1) Clean loss: keep embeddings close to original
            phi_clean = model(images)
            loss_clean = ((phi_clean - phi_org).pow(2).sum(-1)).mean()

            # 2) Adversarial loss: maximize distance
            # Create adversarial images
            images_adv = attack(images, phi_org)
            phi_adv = model(images_adv)

            loss_adv = ((phi_adv - phi_org).pow(2).sum(-1)).mean()

            loss = loss_clean + adv_weight * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.close()

    # Save the fine‑tuned vision encoder
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Checkpoint saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    # single‑GPU training (no distributed)
    run(0, 1, args)