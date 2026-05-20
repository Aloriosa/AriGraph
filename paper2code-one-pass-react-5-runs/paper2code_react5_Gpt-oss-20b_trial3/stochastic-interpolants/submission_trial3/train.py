import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from utils import set_seed, random_mask, get_time_embedding
from models import UNetSmall

def get_dataloader(batch_size, split='train'):
    transform = T.Compose([
        T.ToTensor(),                # [0,1]
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=(split=='train'),
                                          download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'),
                        num_workers=2, pin_memory=True)
    return loader

def train_one_epoch(model, loader, optimizer, device, epoch, lr_scheduler):
    model.train()
    total_loss = 0.0
    for imgs, _ in tqdm(loader, desc=f'Epoch {epoch}', leave=False):
        imgs = imgs.to(device)          # Bx3x32x32
        B = imgs.shape[0]
        mask = random_mask(B, 32, 32, p=0.3).to(device)  # Bx1x32x32

        # Base density: x0 = mask * x1 + (1-mask) * noise
        noise = torch.randn_like(imgs)
        x0 = mask * imgs + (1 - mask) * noise

        # Sample time t
        t = torch.rand(B, 1, 1, 1, device=device)  # Bx1x1x1
        # Interpolant I_t = (1-t)*x0 + t*x1
        I_t = (1 - t) * x0 + t * imgs
        # Derivative: dotI_t = -x0 + x1
        dotI_t = -x0 + imgs

        # Prepare model input: concatenate x_t, mask, t
        t_emb = get_time_embedding(t.squeeze(3).squeeze(2))  # BxC
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)           # BxCx1x1
        # Broadcast t_emb to spatial dims
        t_emb = t_emb.expand(-1, -1, 32, 32)
        inp = torch.cat([I_t, mask, t_emb], dim=1)  # Bx5x32x32

        pred = model(inp)  # Bx3x32x32
        loss = ((pred - dotI_t)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)
        optimizer.step()

        total_loss += loss.item() * B

    avg_loss = total_loss / len(loader.dataset)
    lr_scheduler.step()
    print(f'Epoch {epoch} | Avg loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}')
    return avg_loss

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetSmall().to(device)

    train_loader = get_dataloader(args.batch_size, split='train')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler)

    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    print(f'Model saved to {args.save_model}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save-model', type=str, default='model.pt')
    args = parser.parse_args()
    main(args)