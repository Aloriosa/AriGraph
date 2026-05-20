import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from tqdm import tqdm
import os
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LBCS:
    def __init__(self, dataset_name, k, epsilon, iterations):
        self.dataset_name = dataset_name
        self.k = k
        self.epsilon = epsilon
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.dataset = self.load_dataset()
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=100, shuffle=True)

    def load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if self.dataset_name == "mnist":
            dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        elif self.dataset_name == "fashion_mnist":
            dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        elif self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        return dataset

    def evaluate(self, mask):
        model = LeNet().to(self.device)
        model.load_state_dict(self.model.state_dict())
        model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        return correct / total, loss / len(self.train_loader)

    def run(self):
        n = len(self.dataset)
        mask = np.zeros(n)
        # Initialize mask with random k elements as 1
        mask[:self.k] = 1
        np.random.shuffle(mask)

        best_accuracy = 0
        best_mask = mask.copy()
        best_mask_size = self.k

        for iteration in range(self.iterations):
            # Inner loop: Train model on selected coreset
            train_indices = np.where(mask == 1)[0]
            train_subset = Subset(self.dataset, train_indices)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=100, shuffle=True)

            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Evaluate performance
            accuracy, loss = self.evaluate(mask)
            current_mask_size = np.sum(mask)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Accuracy: {accuracy:.4f}, Coreset size: {current_mask_size}")

            # Lexicographic optimization: first optimize performance, then size
            if accuracy > best_accuracy or (accuracy == best_accuracy and current_mask_size < best_mask_size):
                best_accuracy = accuracy
                best_mask = mask.copy()
                best_mask_size = current_mask_size

            # Update mask based on lexicographic preferences
            # For simplicity, we use a greedy approach to reduce coreset size if performance is acceptable
            if accuracy >= best_accuracy:
                # Try to remove one element from coreset
                for i in range(n):
                    if mask[i] == 1:
                    # Temporarily remove this element
                    mask[i] = 0
                    # Check if performance is still acceptable
                    temp_accuracy, _ = self.evaluate(mask)
                    if temp_accuracy >= best_accuracy - self.epsilon * best_accuracy:
                        # Performance still acceptable, keep the change
                        best_mask = mask.copy()
                    else:
                        # Performance dropped, revert
                    mask[i] = 1

        # Final evaluation
        final_accuracy, final_loss = self.evaluate(best_mask)
        final_mask_size = np.sum(best_mask)

        return final_accuracy, final_mask_size

def main():
    parser = argparse.ArgumentParser(description="Reproduce LBCS algorithm")
    parser.add_argument('--dataset', type=str, default="mnist", help="Dataset to use (mnist, fashion_mnist, cifar10)")
    parser.add_argument('--k', type=int, default=200, help="Initial coreset size")
    parser.add_argument('--epsilon', type=float, default=0.2, help="Performance compromise")
    parser.add_argument('--iterations', type=int, default=500, help="Number of iterations")
    parser.add_argument('--output', type=str, default="output.csv", help="Output file")
    args = parser.parse_args()

    # Run the algorithm
    lbcs = LBCS(dataset_name=args.dataset, k=args.k, epsilon=args.epsilon, iterations=args.iterations)
    final_accuracy, final_mask_size = lbcs.run()

    # Save results
    with open(args.output, 'w') as f:
        f.write(f"final_accuracy,final_mask_size\n")
        f.write(f"{final_accuracy},{final_mask_size}\n")

    print(f"Final accuracy: {final_accuracy:.4f}, Final coreset size: {final_mask_size}")

if __name__ == "__main__":
    main()