import argparse
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from src.utils import set_seed, get_cifar10_dataloaders, save_checkpoint, load_checkpoint
from src.model import DiffusionModel
from src.classifier import BinaryClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cifar10')
    parser.add_argument('--target', type=str, default='cifar10_subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--adaptor-lr', type=float, default=1e-4)
    parser.add_argument('--adversarial-steps', type=int, default=10)
    parser.add_argument('--adversarial-lr', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=5.0)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    source_loader, target_loader, target_indices = get_cifar10_dataloaders(batch_size=args.batch_size, seed=args.seed)
    target_imgs = []
    for imgs, _ in target_loader:
        target_imgs.append(imgs)
    target_imgs = torch.cat(target_imgs, dim=0).to(device)

    # Model
    diff_model = DiffusionModel(device=device).to(device)
    scaler = GradScaler()

    # Classifier
    clf = BinaryClassifier().to(device)
    clf.train_classifier(source_loader, target_loader, epochs=2)

    optimizer = torch.optim.Adam(diff_model.adaptor.parameters(), lr=args.adaptor_lr)

    for epoch in range(args.epochs):
        for imgs, _ in source_loader:
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            timesteps = torch.randint(0, diff_model.scheduler.num_train_timesteps,
                                      (batch_size,), device=device).long()
            noise = torch.randn_like(imgs)
            # adversarial noise selection
            worst_noise = noise.clone()
            for _ in range(args.adversarial_steps):
                worst_noise.requires_grad_()
                with autocast():
                    loss = nn.functional.mse_loss(
                        diff_model.unet(diff_model.scheduler.add_noise(imgs, worst_noise, timesteps), timesteps)[0],
                        worst_noise
                    )
                grad = torch.autograd.grad(loss, worst_noise)[0]
                worst_noise = (worst_noise + args.adversarial_lr * grad).detach()
                # normalize to unit variance
                worst_noise = (worst_noise - worst_noise.mean()) / (worst_noise.std() + 1e-6)
            # forward + loss
            with autocast():
                noisy_imgs = diff_model.scheduler.add_noise(imgs, worst_noise, timesteps)
                pred_noise = diff_model.unet(noisy_imgs, timesteps)[0]
                pred_noise = pred_noise + diff_model.adaptor(noisy_imgs, timesteps)[0]
                # similarity term
                logits = clf(noisy_imgs, timesteps)
                target = torch.ones(logits.shape[0], dtype=torch.long, device=device)
                sim = -args.gamma * (pred_noise * logits[range(logits.shape[0]), target].unsqueeze(1)).mean()
                loss = nn.functional.mse_loss(pred_noise, worst_noise) + sim
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}/{args.epochs} completed.')

    # Save fine‑tuned model
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    save_checkpoint({
        'adaptor_state_dict': diff_model.adaptor.state_dict(),
        'diffusion_state_dict': diff_model.unet.state_dict(),
        'scheduler_state_dict': diff_model.scheduler.state_dict(),
    }, os.path.join(args.output_dir, 'checkpoints', 'fine_tuned.pth'))

    # Save target images for evaluation
    target_save_dir = os.path.join(args.output_dir, 'target_images')
    os.makedirs(target_save_dir, exist_ok=True)
    for i in range(target_imgs.shape[0]):
        img = target_imgs[i]
        img = ((img + 1) / 2 * 255).clamp(0,255).to(torch.uint8)
        img = img.permute(1,2,0).cpu().numpy()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(target_save_dir, f'target_{i:04d}.png'))

if __name__ == '__main__':
    main()