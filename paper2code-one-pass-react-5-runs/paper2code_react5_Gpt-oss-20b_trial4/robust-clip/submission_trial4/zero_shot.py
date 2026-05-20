#!/usr/bin/env python
import argparse
import os
import torch
import clip
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Zero‑shot accuracy on CIFAR‑10")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Fine‑tuned vision encoder checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CIFAR‑10 data.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="CSV file to write results.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original CLIP (text encoder only needed)
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # Load fine‑tuned vision encoder
    model = torch.nn.Sequential(clip_model.visual)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # CIFAR‑10 data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_model.visual.input_mean, std=clip_model.visual.input_std),
    ])
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

    class_names = testset.classes  # ['airplane', 'automobile', ...]

    # Prepare text embeddings
    with torch.no_grad():
        text_tokens = clip_model.tokenize(class_names).to(device)
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        feats = model(imgs)
        feats = feats.float()
        feats /= feats.norm(dim=-1, keepdim=True)

        logits = feats @ text_features.t()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    acc = correct / total * 100
    with open(args.output_file, "w") as f:
        f.write(f"accuracy,{acc:.2f}\n")
    print(f"Zero‑shot CIFAR‑10 accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()