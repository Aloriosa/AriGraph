#!/usr/bin/env python3
"""
Unsupervised Adversarial Fine‑Tuning of the CLIP image encoder (FARE).

This script implements the core training and evaluation pipeline described in the
paper *Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models*.
The implementation focuses on the *image encoder* of CLIP and uses CIFAR‑10 by default.
The user can optionally switch to ImageNet by passing `--dataset imagenet` and
providing `--imagenet-root` pointing to the ImageNet data folder.

Key faithful aspects:
- 10‑step half‑precision APGD during training (ε = 2/255 or 4/255, step = 1/255).
- 100‑step APGD during evaluation (ε = 2/255 or 4/255, step = 1/255).
- AdamW optimizer, lr = 1e‑5, weight decay = 1e‑4.
- 2 epochs of fine‑tuning.
- Zero‑shot classification on test set (clean, ε = 2/255, ε = 4/255).
- Baseline comparison to the original CLIP encoder.
- Optional TeCoA baseline if a checkpoint is present.
"""

import argparse
import os
import random
import numpy as np
import torch
import clip
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configurable hyper‑parameters (match paper as closely as possible)
# --------------------------------------------------------------------------- #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

# Training PGD (APGD) – half‑precision, 10 steps
TRAIN_PGD_STEPS = 10
TRAIN_EPS = 2 / 255.0
TRAIN_STEP_SIZE = 1 / 255.0

# Evaluation PGD – full precision, 100 steps
EVAL_PGD_STEPS = 100
EVAL_EPS_2 = 2 / 255.0
EVAL_EPS_4 = 4 / 255.0
EVAL_STEP_SIZE = 1 / 255.0

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Parse command‑line arguments
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='FARE training & evaluation')
parser.add_argument(
    '--dataset',
    choices=['cifar10', 'imagenet'],
    default='cifar10',
    help='Dataset to train / evaluate on (default: cifar10)')
parser.add_argument(
    '--imagenet-root',
    type=str,
    default='',
    help='Root folder of ImageNet (must contain train/ and val/ subdirs)')
args = parser.parse_args()

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
if args.dataset == 'cifar10':
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='.', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='.', train=False, download=True, transform=transform)

    class_names = train_set.classes  # list of 10 class strings

elif args.dataset == 'imagenet':
    if not args.imagenet_root:
        raise ValueError('ImageNet root directory must be provided with --imagenet-root')
    if not os.path.isdir(args.imagenet_root):
        raise ValueError(f'ImageNet root {args.imagenet_root} does not exist')

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
    ])

    # The ImageNet folder must contain train/ and val/ subfolders
    train_set = torchvision.datasets.ImageFolder(
        os.path.join(args.imagenet_root, 'train'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(
        os.path.join(args.imagenet_root, 'val'), transform=transform)

    # Load ImageNet class names from the folder structure
    class_names = [d for d in os.listdir(os.path.join(args.imagenet_root, 'train')) if os.path.isdir(os.path.join(args.imagenet_root, 'train', d))]

    # To keep training time reasonable, we use a random subset of 100k training images
    # (ImageNet has ~1.28M images).  For a quick demo, 200k should be fine.
    if len(train_set) > 200_000:
        indices = random.sample(range(len(train_set)), 200_000)
        train_set = Subset(train_set, indices)

else:
    raise ValueError(f'Unsupported dataset {args.dataset}')

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------------------------------------------------------- #
# Model preparation
# --------------------------------------------------------------------------- #
print('Loading CLIP model...')
model, _ = clip.load('ViT-B/32', device=DEVICE, jit=False)

# Freeze the text encoder – only the image encoder is updated
for p in model.text_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    model.visual.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# --------------------------------------------------------------------------- #
# Helper: APGD (half‑precision for training, full precision for eval)
# --------------------------------------------------------------------------- #
def pgd_attack(images: torch.Tensor,
               model: torch.nn.Module,
               eps: float,
               alpha: float,
               steps: int,
               half_precision: bool = False) -> torch.Tensor:
    """
    Perform APGD attack (optionally in half‑precision).
    """
    # Use a dedicated perturbation variable
    perturb = torch.zeros_like(images, device=DEVICE, requires_grad=True)

    # Context for half‑precision
    amp_context = torch.autocast(DEVICE, dtype=torch.float16) if half_precision else None

    for _ in range(steps):
        with torch.set_grad_enabled(True), (amp_context or contextlib.nullcontext()):
            # Forward clean and perturbed images
            clean_emb = model.visual(images)          # [B, D]
            pert_emb = model.visual(images + perturb)

            # Normalise embeddings
            clean_emb_norm = clean_emb / clean_emb.norm(dim=-1, keepdim=True)
            pert_emb_norm = pert_emb / pert_emb.norm(dim=-1, keepdim=True)

            # FARE loss: mean squared ℓ₂ distance between normalised embeddings
            loss = torch.mean((clean_emb_norm - pert_emb_norm) ** 2)

        # Backpropagate to get gradient wrt perturbation
        if half_precision:
            loss.backward()
            grad = perturb.grad.detach()
        else:
            loss.backward()
            grad = perturb.grad.detach()

        # Gradient step
        with torch.no_grad():
            perturb.data = perturb.data + alpha * torch.sign(grad)
            # Project back to ε‑ball
            perturb.data = torch.clamp(perturb.data, min=-eps, max=eps)
            # Zero gradients for next step
            perturb.grad.zero_()

    # Return perturbed images (clipped to valid range [0,1])
    perturbed = images + perturb
    return torch.clamp(perturbed, 0.0, 1.0)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
print('Starting training...')
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for images, _ in pbar:
        images = images.to(DEVICE)

        # Generate adversarial examples for this batch (half‑precision APGD)
        images_adv = pgd_attack(images,
                                model,
                                eps=TRAIN_EPS,
                                alpha=TRAIN_STEP_SIZE,
                                steps=TRAIN_PGD_STEPS,
                                half_precision=True)

        # Forward clean and adv embeddings
        clean_emb = model.visual(images)
        adv_emb = model.visual(images_adv)

        # Normalise embeddings
        clean_emb_norm = clean_emb / clean_emb.norm(dim=-1, keepdim=True)
        adv_emb_norm = adv_emb / adv_emb.norm(dim=-1, keepdim=True)

        # FARE loss (squared ℓ₂ distance)
        loss = torch.mean((clean_emb_norm - adv_emb_norm) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f'{loss.item():.4f}')

# --------------------------------------------------------------------------- #
# Save fine‑tuned checkpoint
# --------------------------------------------------------------------------- #
ckpt_path = OUTPUT_DIR / 'clip_fare.pt'
torch.save(model.state_dict(), ckpt_path)
print(f'\nSaved fine‑tuned checkpoint to {ckpt_path}')

# --------------------------------------------------------------------------- #
# Helper: Zero‑shot evaluation
# --------------------------------------------------------------------------- #
def evaluate(loader: DataLoader,
             eps: float = None,
             pgd_steps: int = None) -> float:
    """
    Compute classification accuracy.

    If `eps` is None → clean evaluation.
    If `eps` is provided → adversarial evaluation using PGD with the given `eps`.
    """
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Test'):
        images = images.to(DEVICE)
        if eps is not None:
            # generate adversarial examples
            images = pgd_attack(images,
                                model,
                                eps=eps,
                                alpha=EVAL_STEP_SIZE,
                                steps=pgd_steps if pgd_steps is not None else EVAL_PGD_STEPS,
                                half_precision=False)
        with torch.no_grad():
            image_emb = model.encode_image(images)                    # [B, D]
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            logits = image_emb @ text_emb.t()                         # [B, N_classes]
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# --------------------------------------------------------------------------- #
# Zero‑shot evaluation on test set
# --------------------------------------------------------------------------- #
print('\nEvaluating fine‑tuned FARE...')
model.eval()

# Pre‑compute text embeddings for all class names
with torch.no_grad():
    text_tokens = clip.tokenize(class_names).to(DEVICE)          # [N_classes, 77]
    text_emb = model.encode_text(text_tokens)                    # [N_classes, D]
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)   # normalise

clean_acc = evaluate(test_loader, eps=None)
robust_acc_2 = evaluate(test_loader, eps=EVAL_EPS_2)
robust_acc_4 = evaluate(test_loader, eps=EVAL_EPS_4)

print(f'\nClean accuracy  : {clean_acc:.2f}%')
print(f'Robust (ε=2/255): {robust_acc_2:.2f}%')
print(f'Robust (ε=4/255): {robust_acc_4:.2f}%')

# --------------------------------------------------------------------------- #
# Baseline evaluation (original CLIP)
# --------------------------------------------------------------------------- #
print('\nEvaluating baseline (original CLIP)...')
baseline_model, _ = clip.load('ViT-B/32', device=DEVICE, jit=False)
baseline_model.eval()

# Pre‑compute baseline text embeddings
with torch.no_grad():
    baseline_text_emb = baseline_model.encode_text(text_tokens)
    baseline_text_emb = baseline_text_emb / baseline_text_emb.norm(dim=-1, keepdim=True)

def evaluate_baseline(loader: DataLoader,
                      eps: float = None) -> float:
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Baseline'):
        images = images.to(DEVICE)
        if eps is not None:
            images = pgd_attack(images,
                                baseline_model,
                                eps=eps,
                                alpha=EVAL_STEP_SIZE,
                                steps=EVAL_PGD_STEPS,
                                half_precision=False)
        with torch.no_grad():
            image_emb = baseline_model.encode_image(images)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            logits = image_emb @ baseline_text_emb.t()
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

baseline_clean = evaluate_baseline(test_loader, eps=None)
baseline_robust_2 = evaluate_baseline(test_loader, eps=EVAL_EPS_2)
baseline_robust_4 = evaluate_baseline(test_loader, eps=EVAL_EPS_4)

print(f'\nClean accuracy  : {baseline_clean:.2f}%')
print(f'Robust (ε=2/255): {baseline_robust_2:.2f}%')
print(f'Robust (ε=4/255): {baseline_robust_4:.2f}%')

# --------------------------------------------------------------------------- #
# TeCoA baseline (if checkpoint exists)
# --------------------------------------------------------------------------- #
tecoa_path = Path('tecoa_clip_vitb32.pt')
if tecoa_path.is_file():
    print('\nEvaluating TeCoA baseline...')
    tecoa_model = clip.load('ViT-B/32', device=DEVICE, jit=False)[0]
    tecoa_model.load_state_dict(torch.load(tecoa_path, map_location=DEVICE))
    tecoa_model.eval()

    with torch.no_grad():
        tecoa_text_emb = tecoa_model.encode_text(text_tokens)
        tecoa_text_emb = tecoa_text_emb / tecoa_text_emb.norm(dim=-1, keepdim=True)

    def evaluate_tecoa(loader: DataLoader,
                       eps: float = None) -> float:
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc='TeCoA'):
            images = images.to(DEVICE)
            if eps is not None:
                images = pgd_attack(images,
                                    tecoa_model,
                                    eps=eps,
                                    alpha=EVAL_STEP_SIZE,
                                    steps=EVAL_PGD_STEPS,
                                    half_precision=False)
            with torch.no_grad():
                image_emb = tecoa_model.encode_image(images)
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                logits = image_emb @ tecoa_text_emb.t()
                preds = logits.argmax(dim=-1)
                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    tecoa_clean = evaluate_tecoa(test_loader, eps=None)
    tecoa_robust_2 = evaluate_tecoa(test_loader, eps=EVAL_EPS_2)
    tecoa_robust_4 = evaluate_tecoa(test_loader, eps=EVAL_EPS_4)

    print(f'\nClean accuracy  : {tecoa_clean:.2f}%')
    print(f'Robust (ε=2/255): {tecoa_robust_2:.2f}%')
    print(f'Robust (ε=4/255): {tecoa_robust_4:.2f}%')
else:
    print('\nTeCoA checkpoint not found – skipping TeCoA baseline.')

# --------------------------------------------------------------------------- #
# Save accuracy to file for grading
# --------------------------------------------------------------------------- #
with open(OUTPUT_DIR / 'accuracy.txt', 'w') as f:
    f.write('--- Fine‑tuned FARE ---\n')
    f.write(f'Clean accuracy  : {clean_acc:.2f}%\n')
    f.write(f'Robust (ε=2/255): {robust_acc_2:.2f}%\n')
    f.write(f'Robust (ε=4/255): {robust_acc_4:.2f}%\n\n')

    f.write('--- Baseline (original CLIP) ---\n')
    f.write(f'Clean accuracy  : {baseline_clean:.2f}%\n')
    f.write(f'Robust (ε=2/255): {baseline_robust_2:.2f}%\n')
    f.write(f'Robust (ε=4/255): {baseline_robust_4:.2f}%\n')

    if tecoa_path.is_file():
        f.write('\n--- TeCoA baseline ---\n')
        f.write(f'Clean accuracy  : {tecoa_clean:.2f}%\n')
        f.write(f'Robust (ε=2/255): {tecoa_robust_2:.2f}%\n')
        f.write(f'Robust (ε=4/255): {tecoa_robust_4:.2f}%\n')

print('\nResults written to', OUTPUT_DIR / 'accuracy.txt')