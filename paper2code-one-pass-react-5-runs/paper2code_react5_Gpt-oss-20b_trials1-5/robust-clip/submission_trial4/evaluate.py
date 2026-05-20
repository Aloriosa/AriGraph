#!/usr/bin/env python
import argparse
import os
import torch
import torchattacks
import clip
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="FARE‑CLIP evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine‑tuned vision encoder checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet validation subset.")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split (val).")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="Epsilon for adversarial evaluation.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="CSV file to write results.")
    return parser.parse_args()

def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    return model, preprocess

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original CLIP for reference (text encoder)
    clip_orig, preprocess = load_clip(device)
    clip_orig.eval()

    # Load fine‑tuned vision encoder
    model = torch.nn.Sequential(clip_orig.visual)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Data loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_orig.visual.input_mean, std=clip_orig.visual.input_std),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True)

    # Attack (APGD) if eps > 0
    if args.eps > 0:
        attack = torchattacks.APGD(
            model,
            loss_func=lambda y, y_hat: -((y_hat - y).pow(2).sum(-1)).mean(),
            eps=args.eps,
            alpha=args.eps / 10,
            steps=10,
            rand_init=True,
            rand_init_eps=args.eps,
        )
    else:
        attack = None

    correct = 0
    total = 0
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        if attack is not None:
            imgs_adv = attack(imgs, clip_orig.visual(imgs))
            feats = model(imgs_adv)
        else:
            feats = model(imgs)

        logits = clip_orig.encode_image(imgs)  # use original image embeddings for classification
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    acc = correct / total * 100
    with open(args.output_file, "w") as f:
        f.write(f"epsilon,{args.eps}\n")
        f.write(f"accuracy,{acc:.2f}\n")
    print(f"Evaluation done. Accuracy={acc:.2f}% for eps={args.eps}")

if __name__ == "__main__":
    main()