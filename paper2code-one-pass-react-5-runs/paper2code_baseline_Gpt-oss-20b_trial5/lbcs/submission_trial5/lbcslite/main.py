import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from .utils import get_dataset, create_mask, evaluate_loss, save_mask, save_loss_history, save_coreset_size
from .models import SimpleCNN

def train_masked_model(model, optimizer, data_loader, device, epochs=1):
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
    parser = argparse.ArgumentParser(description="LBCS Core Selection")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name (mnist or cifar10)")
    parser.add_argument("--k", type=int, default=1000, help="Target coreset size")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Relative tolerance for loss")
    parser.add_argument("--iterations", type=int, default=100, help="Number of outer iterations")
    parser.add_argument("--inner_epochs", type=int, default=5, help="Epochs for inner training")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full training dataset
    train_dataset = get_dataset(args.dataset, train=True)
    full_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    n_samples = len(train_dataset)
    # Random initial mask
    init_indices = np.random.choice(n_samples, size=args.k, replace=False)
    best_mask = create_mask(n_samples, init_indices).to(device)
    best_indices = init_indices.copy()

    # Train initial model on the initial mask
    model = SimpleCNN(num_classes=10 if args.dataset=="mnist" else 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    subset = Subset(train_dataset, best_indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    train_masked_model(model, optimizer, subset_loader, device, epochs=args.inner_epochs)
    best_loss = evaluate_loss(model, full_loader, device)

    loss_history = [best_loss]
    print(f"Initial loss: {best_loss:.4f} on full training set with {args.k} samples")

    # Outer loop
    for it in tqdm(range(1, args.iterations + 1), desc="Outer loop"):
        # Sample a new random mask
        new_indices = np.random.choice(n_samples, size=args.k, replace=False)
        new_mask = create_mask(n_samples, new_indices).to(device)

        # Train model on new mask
        new_model = SimpleCNN(num_classes=10 if args.dataset=="mnist" else 10).to(device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
        new_subset = Subset(train_dataset, new_indices)
        new_subset_loader = torch.utils.data.DataLoader(new_subset, batch_size=args.batch_size, shuffle=True)
        train_masked_model(new_model, new_optimizer, new_subset_loader, device, epochs=args.inner_epochs)

        # Evaluate loss on full training set
        new_loss = evaluate_loss(new_model, full_loader, device)

        # Lexicographic comparison
        update = False
        if new_loss < best_loss:
            update = True
        elif abs(new_loss - best_loss) <= args.epsilon * best_loss and args.k < len(best_indices):
            update = True

        if update:
            best_loss = new_loss
            best_indices = new_indices
            best_mask = new_mask

        loss_history.append(best_loss)

    # Save outputs
    save_mask(best_indices, out_dir="output")
    save_loss_history(loss_history, out_dir="output")
    save_coreset_size(len(best_indices), out_dir="output")

    print(f"Finished. Best loss: {best_loss:.4f}, coreset size: {len(best_indices)}")
    print("Results stored in the 'output/' directory.")

if __name__ == "__main__":
    main()