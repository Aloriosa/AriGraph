import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, lr_adapter=0.005, lr_desc=0.01, lr_router=0.005, lr_cls=0.005,
                 weight_decay=0.0):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Separate optimizers for different groups
        self.optimizer = torch.optim.SGD(
            [
                {'params': self.model.classifier.parameters(), 'lr': lr_cls},
                {'params': self.model.get_trainable_params(), 'lr': lr_adapter},
            ],
            momentum=0.9, weight_decay=weight_decay
        )

    def train_task(self, train_loader, task_classes, epoch=5, threshold=1.0):
        """
        Train on a single task.
        :param train_loader: DataLoader for the task
        :param task_classes: list of class indices belonging to this task
        :param epoch: number of epochs to train
        :param threshold: z‑score threshold for self‑expansion
        """
        self.model.train()
        offset = 0  # number of classes seen so far

        # -------- 1. Scan epoch for expansion ----------
        for _ in range(1):
            # Accumulate reconstruction errors per descriptor
            error_sums = [torch.zeros(len(layer.descriptors)) for layer in self.model.layers]
            error_sqs  = [torch.zeros(len(layer.descriptors)) for layer in self.model.layers]
            counts     = [torch.zeros(len(layer.descriptors), dtype=torch.long) for layer in self.model.layers]

            for imgs, labels in tqdm(train_loader, desc='Scanning', leave=False):
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    logits, feats = self.model(imgs, return_feats=True)

                for l, layer in enumerate(self.model.layers):
                    feat = feats[l]  # (B, D)
                    for k, desc in enumerate(layer.descriptors):
                        recon = desc(feat)
                        err = F.mse_loss(recon, feat, reduction='none').mean(dim=1)  # (B,)
                        error_sums[l][k] += err.sum()
                        error_sqs[l][k]  += (err ** 2).sum()
                        counts[l][k]     += err.numel()

            # Compute mean & std per descriptor
            means = [s / c for s, c in zip(error_sums, counts)]
            stds  = [torch.sqrt((sq / c) - (m ** 2)).clamp_min(1e-8)
                     for sq, c, m in zip(error_sqs, counts, means)]

            # Decide which layers to expand
            expand_layers = []
            for l, layer in enumerate(self.model.layers):
                z_scores = [(m - layer.mus[k]) / layer.sigmas[k]
                            for k, m in enumerate(means[l])]
                if all(z > threshold for z in z_scores):
                    expand_layers.append(l)

            # Add adapters/descriptors/router for the chosen layers
            for l in expand_layers:
                layer = self.model.layers[l]
                layer.add_adapter()
                # Initialize new descriptor statistics with current mean/std
                new_k = len(layer.descriptors) - 1
                layer.mus[new_k]  = nn.Parameter(means[l][new_k].clone().detach(), requires_grad=False)
                layer.sigmas[new_k] = nn.Parameter(stds[l][new_k].clone().detach(), requires_grad=False)

        # -------- 2. Training epoch ----------
        for epoch_idx in range(epoch):
            epoch_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f'Train Task (epoch {epoch_idx+1}/{epoch})'):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(imgs)

                # Only compute CE loss on current task classes
                logits_task = logits[:, offset : offset + len(task_classes)]
                target_offset = labels - offset
                loss_ce = self.criterion(logits_task, target_offset)

                # RD loss (sum of reconstruction losses for all descriptors)
                _, feats = self.model(imgs, return_feats=True)
                loss_rd = 0.0
                for l, layer in enumerate(self.model.layers):
                    feat = feats[l]
                    for desc in layer.descriptors:
                        loss_rd += F.mse_loss(desc(feat), feat, reduction='none').mean()

                loss = loss_ce + loss_rd
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f'  Epoch {epoch_idx+1} loss: {avg_loss:.4f}')

        # Update offset for next task
        offset += len(task_classes)
        return

    def evaluate(self, test_loader, seen_classes):
        """Compute accuracy on the union of seen classes."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                logits_task = logits[:, :len(seen_classes)]
                preds = logits_task.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total