# main.py
import torch
import timm
import numpy as np
import math
import os
from foa import FOA
from utils import get_cifar10_loaders, compute_source_stats, ece

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ViT‑Base (ImageNet‑pretrained)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    model.to(device)
    model.eval()

    # Data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    # Compute source statistics using first 32 training images
    source_stats = compute_source_stats(model, train_loader, device, n_samples=32)
    print("Source CLS mean shape:", source_stats["mu"].shape)

    # FOA hyper‑parameters
    prompt_dim = model.patch_embed.out_dim   # 768
    n_prompt = 3
    lambda_reg = 0.4
    gamma = 1.0
    popsize = 28
    cma_iters = 5

    foa = FOA(
        model=model,
        device=device,
        prompt_dim=prompt_dim,
        n_prompt=n_prompt,
        source_stats=source_stats,
        lambda_reg=lambda_reg,
        gamma=gamma,
        popsize=popsize,
        cma_iters=cma_iters,
        verbose=False,
    )

    # Evaluation loop
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="FOA test")):
        images = images.to(device)
        labels = labels.to(device)

        # Adapt prompts for this batch
        prompt = foa.adapt_batch(images)

        # Predict
        with torch.no_grad():
            logits = foa.predict(images, prompt)
            probs = torch.softmax(logits, dim=-1)

            preds = logits.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    acc = 100.0 * total_correct / total_samples
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ece_val = ece(all_probs, all_labels)

    print(f"\nFinal Test Accuracy: {acc:.2f} %")
    print(f"Final Test ECE:      {ece_val:.4f} %")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/foa_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.2f} %\n")
        f.write(f"ECE: {ece_val:.4f} %\n")


if __name__ == "__main__":
    main()