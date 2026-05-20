import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import random
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LeNet(nn.Module):
    """Simplified LeNet architecture for MNIST"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

class LBCS:
    """Lexicographic Bilevel Coreset Selection implementation"""
    def __init__(self, dataset_name='mnist', k=1000, epsilon=0.2, max_iterations=100):
        self.dataset_name = dataset_name
        self.k = k
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.initialize_dataset()
        
    def initialize_dataset(self):
        """Initialize the dataset based on the specified dataset name"""
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
            # Use a subset for faster training
            train_indices = np.random.choice(len(train_dataset), size=min(10000, len(train_dataset)), replace=False)
            test_indices = np.random.choice(len(test_dataset), size=min(1000, len(test_dataset)), replace=False)
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            test_subset = torch.utils.data.Subset(test_dataset, test_indices)
            
            self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
            
            # Store data for coreset selection
            X = []
            y = []
            for i in range(len(train_subset)):
                x, label = train_subset[i]
            # Convert to numpy
            X = np.array(X)
            y = np.array(y)
            self.X_train = X
            self.y_train = y
            
            # Create test data
            X_test = []
            y_test = []
            for i in range(len(test_subset)):
                x, label = test_subset[i]
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            self.X_test = X_test
            self.y_test = y_test
            
        elif self.dataset_name == 'fashion_mnist':
            # Similar to MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            
            train_indices = np.random.choice(len(train_dataset), size=min(10000, len(train_dataset)), replace=False)
            test_indices = np.random.choice(len(test_dataset), size=min(100, len(test_dataset)), replace=False)
            
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            test_subset = torch.utils.data.Subset(test_dataset, test_indices)
            
            self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
            
            # Store data for coreset selection
            X = []
            y = []
            for i in range(len(train_subset)):
                x, label = train_subset[i]
            X = np.array(X)
            y = np.array(y)
            self.X_train = X
            self.y_train = y
            
            # Create test data
            X_test = []
            y_test = []
            for i in range(len(test_subset)):
                x, label = test_subset[i]
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            self.X_test = X_test
            self.y_test = y_test
        else:
            raise ValueError("Dataset not supported")
            
    def train_model(self, masks, epochs=10):
        """Train the model with the given masks"""
        # Initialize model
        self.model = LeNet(num_classes=10).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train the model
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Apply masks
                inputs = inputs * masks.view(-1, 1, 1, 1)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Print loss
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(self.train_loader):.4f}')
            
        return self.model
    
    def evaluate_model(self, model, masks):
        """Evaluate the model on the test set"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Apply masks
            inputs = inputs * masks.view(-1, 1, 1, 1, 1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def lexicographic_optimization(self):
        """Perform lexicographic optimization for RCS"""
        # Initialize masks
        n = len(self.X_train)
        masks = np.random.choice([0, 1], size=n, p=[1 - self.k/n, self.k/n])
        
        # Initialize best performance and coreset size
        best_performance = float('inf')
        best_margins = np.zeros(n)
        
        # Perform lexicographic optimization
        for iteration in range(self.max_iterations):
            # Train model with current masks
            model = self.train_model(masks, epochs=1)
            
            # Evaluate model performance
            performance = self.evaluate_model(model, masks)
            
            # Calculate coreset size
            coreset_size = np.sum(masks)
            
            # Update best performance and coreset size
            if performance < best_performance:
                best_performance = performance
                best_margins = masks.copy()
            
            # Update masks for next iteration
            # Simple random walk for exploration
            if iteration % 5 == 0:
                # Random perturbation
                perturbation = np.random.normal(0, 0.1, n)
                masks = masks + perturbation
                masks = np.clip(masks, 0, 1)
                masks = (masks > 0.5).astype(int)
        
        return best_margins, best_performance
    
    def run(self):
        """Run the LBCS algorithm"""
        print("Starting LBCS algorithm...")
        start_time = time.time()
        
        # Perform lexicographic optimization
        best_margins, best_performance = self.lexicographic_optimization()
        
        # Final evaluation
        final_performance = best_performance
        final_margins = best_margins
        final_size = np.sum(final_margins)
        
        # Print results
        print(f"Final performance: {final_performance:.4f}")
        print(f"Final coreset size: {final_size}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        return final_performance, final_margins, final_size

def main():
    parser = argparse.ArgumentParser(description='Lexicographic Bilevel Coreset Selection')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--k', type=int, default=1000, help='Target coreset size')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Performance compromise')
    parser.add_argument('--output', type=str, default='output.csv', help='Output file')
    args = parser.parse_args()
    
    # Initialize LBCS
    lbc_s = LBCS(dataset_name=args.dataset, k=args.k, epsilon=args.epsilon)
    
    # Run LBCS
    final_performance, final_margins, final_size = lbc_s.run()
    
    # Save results
    np.savetxt(args.output, [final_performance, final_size], delimiter=',')
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()