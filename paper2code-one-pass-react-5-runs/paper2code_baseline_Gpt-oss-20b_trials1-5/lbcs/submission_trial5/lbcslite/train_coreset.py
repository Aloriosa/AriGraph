import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from .utils import get_dataset, load_mask, evaluate_loss, save_accuracy
from .models import SimpleCNN

def train_on_coreset(model, optimizer, data_loader, device, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="Train a model on the selected coreset")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name (mnist or cifar10)")
    parser.add_argument("--mask_file", type=str, required=True, help="Path to the .npy mask file")
    parser.add_argument("--inner_epochs", type=int, default=10, help="Training epochs on the coreset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and mask
    train_dataset = get_dataset(args.dataset, train=True)
    test_dataset = get_dataset(args.dataset, train=False)

    mask_indices = load_mask(args.mask_file)
    subset = torch.utils.data.Subset(train_dataset, mask_indices)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train model on coreset
    model = SimpleCNN(num_classes=10 if args.dataset=="mnist" else 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_on_coreset(model, optimizer, train_loader, device, epochs=args.inner_epochs)

    # Evaluate on test set
    test_loss = evaluate_loss(model, test_loader, device)
    # Compute accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    save_accuracy(accuracy, out_dir="output")

if __name__ == "__main__":
    main()