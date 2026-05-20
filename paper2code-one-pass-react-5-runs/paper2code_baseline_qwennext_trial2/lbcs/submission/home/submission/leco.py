import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import csv

# Define the network architecture (LeNet)
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the function to load the dataset
def load_dataset(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4354, 0.3955), (0.1387, 0.1345, 0.1275))
        ])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return train_dataset, test_dataset

# Define the function to train the network
def train_network(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Define the function to evaluate the network
def evaluate_network(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100 * correct / total

# Define the function to perform coreset selection
def perform_coreset_selection(model, train_dataset, k, epsilon, device):
    # Randomly sample k points from the training set
    indices = np.random.choice(len(train_dataset), size=k, replace=False)
    coreset = torch.utils.data.Subset(train_dataset, indices)
    return coreset

# Define the function to save the results
def save_results(results, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'k', 'epsilon', 'accuracy', 'coreset_size'])
        for result in results:
            writer.writerow(result)

# Define the main function
def main():
    parser = argparse.ArgumentParser(description='Refined Coreset Selection')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--k', type=int, default=1000, help='size of the coreset')
    parser.add_argument('--epsilon', type=float, default=0.2, help='performance compromise')
    parser.add_argument('--output', type=str, default='output.csv', help='output file')
    args = parser.parse_args()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_dataset, test_dataset = load_dataset(args.dataset)

    # Create the model
    model = LeNet().to(device)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils_data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Train the model on the full dataset
    train_network(model, train_loader, device, epochs=10, lr=0.001)

    # Perform coreset selection
    coreset = perform_coreset_selection(model, train_dataset, args.k, args.epsilon, device)

    # Train the model on the coreset
    coreset_loader = torch.utils.data.DataLoader(coreset, batch_size=100, shuffle=True)
    train_network(model, coreset_loader, device, epochs=10, lr=0.001)

    # Evaluate the model on the test set
    accuracy = evaluate_network(model, test_loader, device)

    # Save the results
    results = [[args.dataset, args.k, args.epsilon, accuracy, len(coreset)]]
    save_results(results, args.output)

    # Print the results
    print(f"Dataset: {args.dataset}")
    print(f"K: {args.k}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Coreset Size: {len(coreset)}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()