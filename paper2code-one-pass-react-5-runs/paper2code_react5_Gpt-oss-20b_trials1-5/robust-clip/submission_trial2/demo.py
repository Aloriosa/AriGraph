#!/usr/bin/env python3
"""
Zero‑shot classification demo for original CLIP and FARE‑CLIP.
Also shows the effect of a PGD attack on a single image.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from datasets import load_dataset

from apgd_attack import apgd_attack

# ---------------------------------------------------------------------------

def preprocess_cifar10(img, preprocess):
    """Convert a PIL image to the CLIP tensor format."""
    return preprocess(img).unsqueeze(0)  # shape [1,3,224,224] after preprocess

# ---------------------------------------------------------------------------

def zero_shot_accuracy(
    clip_model,
    classes,
    dataloader,
    device,
    eps=None,
    steps=50,
    step_size=2/255,
):
    """
    Compute zero‑shot accuracy on the given DataLoader.
    If eps>0, images are attacked before embedding.
    """
    clip_model.eval()
    all_logits = []
    all_labels = []

    # Build text embeddings for all classes
    with torch.no_grad():
        text_tokens = clip.tokenize(classes).to(device)  # [C, T]
        text_emb = clip_model.encode_text(text_tokens)  # [C, D]
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    for imgs, labels in tqdm(dataloader, desc="Eval batch"):
        imgs = imgs.to(device)
        if eps:
            imgs = apgd_attack(
                clip_model,
                imgs,
                epsilon=eps,
                steps=steps,
                step_size=step_size,
                device=device,
            )
        with torch.no_grad():
            img_emb = clip_model.encode_image(imgs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            logits = img_emb @ text_emb.t()  # [B, C]
            preds = logits.argmax(dim=-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc = (all_logits.argmax(dim=-1) == all_labels).float().mean().item()
    return acc

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero‑shot demo")
    parser.add_argument("--clip_model", type=str,
                        default="openai/clip-vit-base-patch32",
                        help="Model name for original CLIP")
    parser.add_argument("--fare_ckpt", type=str,
                        required=True,
                        help="Path to FARE vision encoder checkpoint")
    parser.add_argument("--output_dir", type=str, default="demo",
                        help="Directory to write results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP
    clip_orig, preprocess = clip.load(args.clip_model, device=device)

    # Load FARE vision encoder and wrap it into a CLIPModel with frozen text encoder
    clip_fare = clip.load(args.clip_model, device=device)[0]
    clip_fare.visual.load_state_dict(torch.load(args.fare_ckpt, map_location=device))
    clip_fare.visual.eval()
    clip_fare.eval()

    # Prepare CIFAR‑10 validation set
    dataset = load_dataset("cifar10", split="test")
    class_names = dataset.features["label"].names

    def collate_fn(batch):
        imgs = [preprocess_cifar10(item["image"], preprocess) for item in batch]
        imgs = torch.cat(imgs, dim=0)  # [B,3,224,224]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return imgs, labels

    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    # 1. Zero‑shot accuracy (clean)
    acc_orig = zero_shot_accuracy(clip_orig, class_names, loader, device)
    acc_fare = zero_shot_accuracy(clip_fare, class_names, loader, device)

    # 2. Zero‑shot accuracy (adversarial)
    eps = 2/255
    acc_orig_adv = zero_shot_accuracy(clip_orig, class_names, loader, device,
                                      eps=eps)
    acc_fare_adv = zero_shot_accuracy(clip_fare, class_names, loader, device,
                                      eps=eps)

    results = {
        "clean": {"original": acc_orig, "fare": acc_fare},
        "adversarial": {"original": acc_orig_adv, "fare": acc_fare_adv},
    }
    with open(os.path.join(args.output_dir, "zero_shot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Zero‑shot results written to", os.path.join(args.output_dir, "zero_shot_results.json"))

    # 3. Show similarity on a single image
    sample = dataset[0]
    img = preprocess_cifar10(sample["image"], preprocess).to(device)

    with torch.no_grad():
        orig_emb = clip_orig.encode_image(img)
        fare_emb = clip_fare.encode_image(img)
        # Attack a single image
        adv_img = apgd_attack(clip_orig, img, epsilon=eps, steps=50,
                              step_size=2/255, device=device)
        adv_emb = clip_orig.encode_image(adv_img)

    def cosine(a, b):
        return F.cosine_similarity(a, b, dim=-1).item()

    with open(os.path.join(args.output_dir, "adversarial.txt"), "w") as f:
        f.write(f"Clean cosine (orig vs fare): {cosine(orig_emb, fare_emb):.4f}\n")
        f.write(f"Clean cosine (orig vs orig): {cosine(orig_emb, orig_emb):.4f}\n")
        f.write(f"Adversarial cosine (orig vs orig): {cosine(adv_emb, orig_emb):.4f}\n")
        f.write(f"Adversarial cosine (fare vs orig): {cosine(adv_emb, fare_emb):.4f}\n")
    print("Adversarial similarity statistics written to", os.path.join(args.output_dir, "adversarial.txt"))

if __name__ == "__main__":
    main()