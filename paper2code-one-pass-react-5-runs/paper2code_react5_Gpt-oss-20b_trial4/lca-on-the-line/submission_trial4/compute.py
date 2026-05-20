#!/usr/bin/env python3
"""
Main evaluation script that:

* Loads pretrained vision and vision‑language models.
* Evaluates them on ImageNet and its OOD variants.
* Computes top‑1/top‑5 accuracy and average LCA distance (over mispredictions).
* Calculates Pearson and R² correlation between ID LCA distance and OOD top‑1 accuracy.
* Optionally builds a latent class hierarchy (via hierarchical K‑means) and uses it
  for LCA computation.
"""

import argparse
import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: import clip only if needed
try:
    import clip
except Exception:
    clip = None

from utils.imagenet_utils import get_imagenet_class_mappings
from utils.lca import (
    compute_average_lca,
    compute_average_lca_matrix,
    compute_pearson_r2,
    compute_info_content,
)
from utils.latent_lca import build_latent_hierarchy, build_lca_matrix

# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------
# Each entry: (display_name, constructor_function)
MODEL_REGISTRY = [
    ("resnet18", lambda: torchvision.models.resnet18(pretrained=True)),
    ("resnet50", lambda: torchvision.models.resnet50(pretrained=True)),
    # CLIP from OpenAI
    ("clip-vit-base-patch32", "clip"),
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_imagenet_split(split_dir, batch_size=64):
    """
    Load an ImageNet split from a directory containing the
    standard 'val' folder structure:
        split_dir/
            val/
                nxxxxxx/
                    img1.jpg
                    ...
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if not os.path.isdir(split_dir):
        print(f"Directory {split_dir} not found.", file=sys.stderr)
        sys.exit(1)

    dataset = torchvision.datasets.ImageFolder(os.path.join(split_dir, "val"), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4, pin_memory=True)
    return loader, dataset


def evaluate_model(
    model,
    loader,
    synsets,
    info_content_dict,
    device="cuda",
    use_clip=False,
    clip_model=None,
    clip_text_emb=None,
    lca_matrix=None,
):
    """
    Evaluate a model on the given loader.
    Returns top‑1 accuracy, top‑5 accuracy, and average LCA distance.
    """
    if not use_clip:
        model = model.to(device)
        model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    lca_sum = 0.0
    mispred_count = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            if use_clip:
                # CLIP requires image embeddings and text embeddings
                img_emb = clip_model.encode_image(images)  # [B, dim]
                logits = img_emb @ clip_text_emb.T  # [B, num_classes]
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                top5 = probs.topk(5, dim=1).indices
            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                top5 = probs.topk(5, dim=1).indices

            correct_top1 += (preds == targets).sum().item()
            correct_top5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

            # LCA for mispredictions only
            for pred, tgt in zip(preds.cpu().numpy(), targets.cpu().numpy()):
                if pred != tgt:
                    if lca_matrix is None:
                        lca_sum += compute_average_lca([pred], [tgt], synsets, info_content_dict)
                    else:
                        lca_sum += compute_average_lca_matrix([pred], [tgt], lca_matrix)
                    mispred_count += 1

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    avg_lca = lca_sum / mispred_count if mispred_count > 0 else 0.0
    return top1_acc, top5_acc, avg_lca


def build_clip_text_embeddings(model_name, synsets, device="cuda"):
    """
    Pre‑compute text embeddings for all ImageNet classes for zero‑shot CLIP.
    """
    if clip is None:
        raise RuntimeError("clip module not available")
    # load the same CLIP model that was used for inference
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    class_names = [syn.lemmas()[0].name() for syn in synsets]
    prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)  # [num_classes, dim]
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
    return text_emb


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LCA evaluation")
    parser.add_argument("--imagenet-dir", type=str, required=True,
                        help="Path to ImageNet root directory (contains val/)")
    parser.add_argument("--imagenet-v2-dir", type=str, default="",
                        help="Optional ImageNet‑v2 root directory (contains val/)")
    parser.add_argument("--ood-dirs", nargs="*", default=[],
                        help="Paths to OOD split directories (each containing val/)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--latent-hierarchy", action="store_true",
                        help="Build and use a latent class hierarchy")
    parser.add_argument("--max-levels", type=int, default=9,
                        help="Maximum levels for latent hierarchy")
    parser.add_argument("--correlate", action="store_true",
                        help="Compute Pearson and R² correlation between ID LCA and OOD top‑1")
    args = parser.parse_args()

    # Load ImageNet validation split
    id_loader, id_dataset = load_imagenet_split(args.imagenet_dir, args.batch_size)
    synsets = get_imagenet_class_mappings(id_dataset)

    # Pre‑compute information‑content for each synset
    info_content_dict = compute_info_content(synsets)

    # Optionally build latent hierarchy
    lca_matrix = None
    if args.latent_hierarchy:
        print("Building latent hierarchy (this may take a few minutes)...")
        # Extract per‑class average features using ResNet50
        resnet = torchvision.models.resnet50(pretrained=True).to(args.device)
        resnet.eval()
        feat_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        class_feats = []
        with torch.no_grad():
            for cls_idx in range(len(id_dataset.classes)):
                imgs = []
                for _ in range(10):  # average over 10 random images
                    idx = np.random.randint(len(id_dataset))
                    img, _ = id_dataset[idx]
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs).to(args.device)
                features = feat_extractor(imgs).reshape(len(imgs), -1).mean(dim=0)
                class_feats.append(features.cpu().numpy())
        class_feats = np.stack(class_feats)
        assignments = build_latent_hierarchy(class_feats, max_levels=args.max_levels)
        lca_matrix = build_lca_matrix(assignments)

    results = []

    for name, constructor in MODEL_REGISTRY:
        print(f"\n=== Evaluating {name} ===")
        if constructor == "clip":
            # Load CLIP model (CPU or GPU)
            clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
            model = None
            use_clip = True
            clip_text_emb = build_clip_text_embeddings(name, synsets, device=args.device)
        else:
            model = constructor()
            use_clip = False
            clip_text_emb = None

        # Evaluate on ID split
        id_top1, id_top5, id_lca = evaluate_model(
            model,
            id_loader,
            synsets,
            info_content_dict,
            device=args.device,
            use_clip=use_clip,
            clip_model=clip_model if use_clip else None,
            clip_text_emb=clip_text_emb,
            lca_matrix=lca_matrix,
        )
        print(f"ID   Top‑1 Acc: {id_top1:.4f} | Top‑5 Acc: {id_top5:.4f} | Avg LCA: {id_lca:.4f}")

        record = {
            "model": name,
            "imagenet_top1": id_top1,
            "imagenet_top5": id_top5,
            "imagenet_lca": id_lca,
        }

        # Evaluate on ImageNet‑v2 if provided
        if args.imagenet_v2_dir:
            v2_loader, _ = load_imagenet_split(args.imagenet_v2_dir, args.batch_size)
            v2_top1, v2_top5, v2_lca = evaluate_model(
                model,
                v2_loader,
                synsets,
                info_content_dict,
                device=args.device,
                use_clip=use_clip,
                clip_model=clip_model if use_clip else None,
                clip_text_emb=clip_text_emb,
                lca_matrix=lca_matrix,
            )
            print(f"v2   Top‑1 Acc: {v2_top1:.4f} | Top‑5 Acc: {v2_top5:.4f} | Avg LCA: {v2_lca:.4f}")
            record["imagenet_v2_top1"] = v2_top1
            record["imagenet_v2_top5"] = v2_top5
            record["imagenet_v2_lca"] = v2_lca

        # Evaluate on each OOD split
        for ood_dir in args.ood_dirs:
            ood_name = os.path.basename(os.path.normpath(ood_dir))
            ood_loader, _ = load_imagenet_split(ood_dir, args.batch_size)
            ood_top1, ood_top5, ood_lca = evaluate_model(
                model,
                ood_loader,
                synsets,
                info_content_dict,
                device=args.device,
                use_clip=use_clip,
                clip_model=clip_model if use_clip else None,
                clip_text_emb=clip_text_emb,
                lca_matrix=lca_matrix,
            )
            print(f"{ood_name} Top‑1 Acc: {ood_top1:.4f} | Top‑5 Acc: {ood_top5:.4f} | Avg LCA: {ood_lca:.4f}")
            record[f"{ood_name}_top1"] = ood_top1
            record[f"{ood_name}_top5"] = ood_top5
            record[f"{ood_name}_lca"] = ood_lca

        results.append(record)

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\nResults written to results.csv")

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------
    if args.correlate:
        corr_rows = []
        ood_keys = [k for k in df.columns if k.endswith("_top1") and k != "imagenet_top1"]
        for key in ood_keys:
            ood_name = key.replace("_top1", "")
            pearson, r2 = compute_pearson_r2(df["imagenet_lca"], df[key])
            corr_rows.append({"dataset": ood_name, "pearson": pearson, "r2": r2})
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv("correlation.csv", index=False)
        print("\nCorrelation statistics written to correlation.csv")


if __name__ == "__main__":
    main()