import argparse
import os
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageNetC, ImageNetVal
from foa import FOAAdapter
from tqdm.auto import tqdm
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="FOA Evaluation")
    parser.add_argument("--datasets", nargs="+", default=["imagenet_c", "imagenet_val"],
                        help="List of datasets (order: test, source)")
    parser.add_argument("--model", default="vit_base_patch16_224", type=str,
                        help="timm model name")
    parser.add_argument("--quantize", default=32, type=int,
                        help="Quantisation level (32 or 8)")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--population-size", default=28, type=int)
    parser.add_argument("--prompt-size", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def get_transforms(quantize: int):
    if quantize == 32:
        transform = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.functional.interpolate,
        )
    # For simplicity we keep the same transforms
    return torch.nn.Identity()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    test_root = os.path.join(os.getcwd(), args.datasets[0])
    source_root = os.path.join(os.getcwd(), args.datasets[1])

    test_ds = ImageNetC(test_root, transform=timm.data.create_transform(
        input_size=224, is_training=False, mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]))
    source_ds = ImageNetVal(source_root, transform=timm.data.create_transform(
        input_size=224, is_training=False, mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    source_loader = DataLoader(source_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

    # Build FOA adapter
    foa = FOAAdapter(
        model_name=args.model,
        prompt_len=args.prompt_size,
        batch_size=args.batch_size,
        population_size=args.population_size,
        device=device,
        seed=args.seed,
        lambda_discrepancy=0.3,
        num_id_samples=32,
        ema_alpha=0.1
    )

    # Prepare source statistics
    print("Computing source statistics...")
    foa.prepare_source_statistics(source_loader)

    # If quantise to 8‑bit we use ptq4vit
    if args.quantize == 8:
        try:
            import ptq4vit
            from ptq4vit.quantizer import Quantizer
        except ImportError:
            print("ptq4vit not installed; skipping quantisation.")
        else:
            print("Quantising model to 8‑bit...")
            quantizer = Quantizer(foa.model, 8)
            foa.model = quantizer.quantized_model
            foa.model.eval()

    # Run evaluation
    print("Evaluating FOA on ImageNet‑C")
    acc, ece = foa.evaluate(test_loader)
    print(f"FOA Accuracy: {acc:.2f}%, ECE: {ece:.2f}%")

    # Save predictions for later inspection
    torch.save({"acc": acc, "ece": ece}, "predictions.pt")


if __name__ == "__main__":
    main()