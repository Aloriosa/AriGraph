import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm

from data.cifar100_incr import get_cifar100_incremental, get_transform
from models.sema import SEMA
from utils import accuracy, compute_stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    device = torch.device(cfg['device'])
    set_seed(cfg['seed'])

    # Create base ViT model
    base = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    base = base.to(device)

    # Determine dim of ViT (patch_embed output)
    dim = base.embed_dim

    # Initialise SEMA
    sema = SEMA(base,
                dim=dim,
                hidden_dim=cfg['rd_hidden_dim'],
                expand_layers=cfg['expand_layers'],
                num_classes=0).to(device)

    # Optimizers
    # We will create optimizer groups on the fly per task
    best_acc = 0.0
    all_seen_classes = 0

    # Stats for RD
    mu = {l: torch.zeros(1).to(device) for l in cfg['expand_layers']}
    sigma = {l: torch.ones(1).to(device) for l in cfg['expand_layers']}

    # Training loop over tasks
    for task_id in range(cfg['num_tasks']):
        print(f"\n=== Task {task_id+1}/{cfg['num_tasks']} ===")
        # Data loaders
        train_set = get_cifar100_incremental(
            task_id, cfg['classes_per_task'], cfg['num_tasks'],
            train=True, transform=get_transform(train=True))
        test_set = get_cifar100_incremental(
            task_id, cfg['classes_per_task'], cfg['num_tasks'],
            train=False, transform=get_transform(train=False))

        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4, pin_memory=True)

        # Expand head for new classes
        sema.add_class(cfg['classes_per_task'])
        all_seen_classes = sema.head.out_features

        # ---------- Phase 1 : Scan for expansion ----------
        print("Scanning for expansion (first epoch)...")
        with torch.no_grad():
            rd_losses = {l: [] for l in cfg['expand_layers']}
            for batch in tqdm(train_loader, desc="Scanning"):
                imgs, labels = batch
                imgs = imgs.to(device)
                _, cls_token = sema(imgs)
                for l in cfg['expand_layers']:
                    for rd in sema.rds[l]:
                        rd_losses[l].append(rd.loss(cls_token))

        # Compute mean/std per RD
        for l in cfg['expand_layers']:
            mu[l] = torch.tensor(np.mean(rd_losses[l])).to(device)
            sigma[l] = torch.tensor(np.std(rd_losses[l]) + 1e-6).to(device)

        # Decide expansion
        expanded = sema.expand_if_needed(
            cls_token, mu, sigma, cfg['expansion_threshold'])
        if expanded:
            print(f"Expansion performed at some layers.")
        else:
            print(f"No expansion needed for this task.")

        # ---------- Phase 2 : Train adapters & RDs ----------
        # Optimizer groups: new adapters & new RDs only
        new_params = []
        for l in cfg['expand_layers']:
            # new adapters are the last ones in the list
            new_params.append(sema.adapters[l][-1].parameters())
            new_params.append(sema.rds[l][-1].parameters())
        # Also train the head
        new_params.append(sema.head.parameters())

        optimizer = optim.AdamW(
            [p for group in new_params for p in group],
            lr=cfg['lr_adapter'],
            weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['num_epochs'])

        for epoch in range(cfg['num_epochs']):
            sema.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits, cls_token = sema(imgs)
                loss_cls = F.cross_entropy(logits, labels)
                loss_rd = sema.get_rd_losses(cls_token)
                loss = loss_cls + loss_rd
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            scheduler.step()
            print(f"  Epoch {epoch+1} loss: {epoch_loss / len(train_loader):.4f}")

        # ---------- Phase 3 : Evaluate ----------
        sema.eval()
        accs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits, _ = sema(imgs)
                accs.append(accuracy(logits, labels, topk=(1,))[0].item())
        acc = np.mean(accs)
        print(f"Task {task_id+1} accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
        print(f"Best accuracy so far: {best_acc:.2f}%")

    print("\n=== Final results ===")
    print(f"Average accuracy over tasks: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)