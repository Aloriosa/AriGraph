import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from typing import Tuple

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    criterion=nn.CrossEntropyLoss()):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             criterion=nn.CrossEntropyLoss()):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def train_and_eval(model: nn.Module,
                   train_loader: DataLoader,
                   test_loader: DataLoader,
                   epochs: int,
                   lr: float,
                   device: torch.device):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train loss: {loss:.4f} | "
              f"Test loss: {test_loss:.4f} | Acc: {acc*100:.2f}%")
    return test_loss, acc