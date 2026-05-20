#!/usr/bin/env python3
"""
Zero‑shot classification evaluation on CIFAR‑10 using the fine‑tuned CLIP encoder.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
import numpy as np

def zero_shot_accuracy(encoder, processor, loader, device):
    """
    Compute zero‑shot accuracy on a DataLoader.
    """
    encoder.eval()
    # Prepare text embeddings for CIFAR‑10 classes
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    text_inputs = processor(text=class_names, padding=True, return_tensors='pt')
    with torch.no_grad():
        text_emb = processor.tokenizer(
            class_names, padding=True, return_tensors='pt').input_ids.to(device)

    # Compute text embeddings once
    text_features = encoder.text_model(
        input_ids=text_emb).last_hidden_state[:, 0, :].to(device)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            img_features = encoder(imgs).last_hidden_state[:, 0, :]
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            logits = (img_features @ text_features.T).cpu()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load fine‑tuned encoder
    encoder = torch.hub.load('pytorch/vision:v0.17.0', 'clip_vit_base_patch32')
    encoder.load_state_dict(torch.load('models/fare_clip.pt', map_location=device))
    encoder.to(device)

    # Processor (tokenizer + image transforms)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Data loader for CIFAR‑10 test set
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    test_set = torchvision.datasets.CIFAR10(root='data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    test_loader = DataLoader(test_set, batch_size=128,
                             shuffle=False, num_workers=4)

    acc = zero_shot_accuracy(encoder, processor, test_loader, device)
    print(f"Zero‑shot accuracy on CIFAR‑10: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()