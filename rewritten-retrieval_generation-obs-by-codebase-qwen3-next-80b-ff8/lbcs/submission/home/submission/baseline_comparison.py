import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import time
import logging
from tqdm import tqdm
from models import ConvNetCIFAR, ResNet, BasicBlock
from datasets_utils.cifar10 import CIFAR10
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineComparison:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = args.dataset
        self.noise_rate = args.noise_rate
        self.coreset_size = args.coreset_size
        self.tolerance = args.tolerance
        self.save_dir = args.save_dir
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Load datasets
        self.train_dataset, self.test_dataset, self.val_dataset = self._load_datasets()
        
        # Get full dataset performance baseline
        self.full_data_accuracy = self._compute_full_data_baseline()
        
        # Define performance constraint
        if '%' in self.tolerance:
            tolerance_percent = float(self.tolerance.replace('%', ''))
            self.performance_threshold = self.full_data_accuracy * (1 - tolerance_percent / 100)
        else:
            self.performance_threshold = self.full_data_accuracy - float(self.tolerance)
        
        logger.info(f"Full data baseline accuracy: {self.full_data_accuracy:.2f}%")
        logger.info(f"Performance threshold: {self.performance_threshold:.2f}%")
        
        # Define model architecture
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Define baselines to compare
        self.baselines = [
            "uniform",
            "el2n",
            "grand",
            "influential",
            "moderate",
            "ccs",
            "probabilistic"
        ]
        
        # Store results
        self.results = {}
    
    def _load_datasets(self):
        """Load and prepare datasets with noise"""
        if self.dataset_name == "fashion_mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Load noisy labels
            noisy_targets = np.load(f'tmp/fashion_noisy_target.npy')
            train_dataset.targets = torch.from_numpy(noisy_targets)
            
            # Create validation set (10% of training data)
            num_train = len(train_dataset)
            num_val = int(0.1 * num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[num_val:], indices[:num_val]
            
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            return train_subset, test_dataset, val_subset
            
        elif self.dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            train_dataset = CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Load noisy labels
            noisy_targets = np.load(f'tmp/cifar10_noisy_target.npy')
            train_dataset.targets = noisy_targets
            
            # Create validation set (10% of training data)
            num_train = len(train_dataset)
            num_val = int(0.1 * num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[num_val:], indices[:num_val]
            
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            return train_subset, test_dataset, val_subset
            
        elif self.dataset_name == "svhn":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
            
            train_dataset = datasets.SVHN(
                root='./data', split='train', download=True, transform=transform
            )
            test_dataset = datasets.SVHN(
                root='./data', split='test', download=True, transform=transform
            )
            
            # Load noisy labels
            noisy_targets = np.load(f'tmp/svhn_noisy_target.npy')
            train_dataset.labels = noisy_targets
            
            # Create validation set (10% of training data)
            num_train = len(train_dataset)
            num_val = int(0.1 * num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[num_val:], indices[:num_val]
            
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            return train_subset, test_dataset, val_subset
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_model(self):
        """Create model based on dataset"""
        if self.dataset_name == "fashion_mnist":
            return ConvNetCIFAR(
                channel=1, 
                num_classes=10, 
                net_width=128, 
                net_depth=2, 
                net_act='relu', 
                net_norm='batchnorm', 
                net_pooling='avgpooling'
            )
        elif self.dataset_name == "cifar10":
            return ConvNetCIFAR(
                channel=3, 
                num_classes=10, 
                net_width=128, 
                net_depth=2, 
                net_act='relu', 
                net_norm='batchnorm', 
                net_pooling='avgpooling'
            )
        elif self.dataset_name == "svhn":
            # Use ResNet for SVHN
            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _compute_full_data_baseline(self):
        """Compute baseline accuracy on full training data"""
        logger.info("Computing baseline accuracy on full training data...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Train model on full data for a few epochs
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):
            total_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target, _) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")
        
        # Evaluate on validation set
        model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        logger.info(f"Full data baseline accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _train_and_evaluate(self, coreset_indices, method_name):
        """Train model on coreset and evaluate on validation set"""
        # Create coreset dataset
        coreset_dataset = Subset(self.train_dataset, coreset_indices)
        coreset_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model.train()
        for epoch in range(100):
            total_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target, _) in enumerate(coreset_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            if epoch % 20 == 0:
                logger.info(f"{method_name} - Epoch {epoch+1}/100, Loss: {total_loss/len(coreset_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy, len(coreset_indices)
    
    def _select_coreset_uniform(self):
        """Uniform random sampling"""
        logger.info("Selecting coreset with uniform sampling...")
        indices = np.random.choice(len(self.train_dataset), size=self.coreset_size, replace=False)
        return indices
    
    def _select_coreset_el2n(self):
        """EL2N (Expected Loss 2 Norm) sampling"""
        logger.info("Selecting coreset with EL2N sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Compute EL2N scores
        el2n_scores = []
        model.eval()
        with torch.no_grad():
            for data, target, indices in tqdm(train_loader, desc="Computing EL2N scores"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                # EL2N = L2 norm of loss
                el2n_scores.extend(loss.cpu().numpy())
        
        # Sort by EL2N scores (higher is more important)
        sorted_indices = np.argsort(el2n_scores)[::-1]
        return sorted_indices[:self.coreset_size]
    
    def _select_coreset_grand(self):
        """GRAND (Gradient Norm) sampling"""
        logger.info("Selecting coreset with GRAND sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Compute gradient norms
        gradient_norms = []
        model.train()
        for data, target, indices in tqdm(train_loader, desc="Computing GRAND scores"):
            data, target = data.to(self.device), target.to(self.device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute gradient norm for each sample
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.view(param.grad.size(0), -1), dim=1)
                    gradient_norms.extend(grad_norm.cpu().numpy())
        
        # Sort by gradient norm (higher is more important)
        sorted_indices = np.argsort(gradient_norms)[::-1]
        return sorted_indices[:self.coreset_size]
    
    def _select_coreset_influential(self):
        """Influential sampling (approximate influence function)"""
        logger.info("Selecting coreset with influential sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Compute approximate influence scores
        influence_scores = []
        model.eval()
        with torch.no_grad():
            for data, target, indices in tqdm(train_loader, desc="Computing influence scores"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                # Approximate influence as loss * gradient norm
                influence_scores.extend(loss.cpu().numpy())
        
        # Sort by influence scores (higher is more important)
        sorted_indices = np.argsort(influence_scores)[::-1]
        return sorted_indices[:self.coreset_size]
    
    def _select_coreset_moderate(self):
        """Moderate sampling (based on paper)"""
        logger.info("Selecting coreset with moderate sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Compute moderate scores (combination of loss and gradient)
        moderate_scores = []
        model.train()
        for data, target, indices in tqdm(train_loader, desc="Computing moderate scores"):
            data, target = data.to(self.device), target.to(self.device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Moderate score = loss * gradient norm
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.view(param.grad.size(0), -1), dim=1)
                    loss_values = loss.cpu().numpy()
                    moderate_scores.extend((loss_values * grad_norm.cpu().numpy()).tolist())
        
        # Sort by moderate scores (higher is more important)
        sorted_indices = np.argsort(moderate_scores)[::-1]
        return sorted_indices[:self.coreset_size]
    
    def _select_coreset_ccs(self):
        """Coverage-Centric Coreset (CCS) sampling"""
        logger.info("Selecting coreset with CCS sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        
        # Extract features using the model
        features = []
        model.eval()
        with torch.no_grad():
            for data, target, indices in tqdm(train_loader, desc="Extracting features"):
                data = data.to(self.device)
                # Use feature extractor (remove final layer)
                features_extractor = nn.Sequential(*list(model.children())[:-1])
                feature = features_extractor(data)
                feature = feature.view(feature.size(0), -1)
                features.append(feature.cpu().numpy())
        
        # Combine features
        features = np.concatenate(features, axis=0)
        
        # Use greedy algorithm to select diverse samples
        selected_indices = []
        remaining_indices = list(range(len(features)))
        
        # Start with random sample
        if len(remaining_indices) > 0:
            selected_indices.append(remaining_indices.pop(np.random.randint(len(remaining_indices))))
        
        # Greedily select samples that maximize coverage
        while len(selected_indices) < self.coreset_size and len(remaining_indices) > 0:
            best_idx = None
            best_score = -1
            
            for idx in remaining_indices:
                # Compute minimum distance to any selected sample
                distances = np.linalg.norm(features[idx] - features[selected_indices], axis=1)
                min_distance = np.min(distances)
                
                if min_distance > best_score:
                    best_score = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return np.array(selected_indices)
    
    def _select_coreset_probabilistic(self):
        """Probabilistic sampling (based on paper)"""
        logger.info("Selecting coreset with probabilistic sampling...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Compute probabilities based on loss
        probabilities = []
        model.eval()
        with torch.no_grad():
            for data, target, indices in tqdm(train_loader, desc="Computing probabilities"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                probabilities.extend(loss.cpu().numpy())
        
        # Convert to probabilities (higher loss = higher probability)
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample with replacement
        selected_indices = np.random.choice(
            len(self.train_dataset), 
            size=self.coreset_size, 
            replace=False, 
            p=probabilities
        )
        
        return selected_indices
    
    def run_comparison(self):
        """Run comparison with all baselines"""
        logger.info("Starting baseline comparison...")
        
        # Select coresets for all baselines
        for baseline in self.baselines:
            logger.info(f"Selecting coreset with {baseline}...")
            
            if baseline == "uniform":
                coreset_indices = self._select_coreset_uniform()
            elif baseline == "el2n":
                coreset_indices = self._select_coreset_el2n()
            elif baseline == "grand":
                coreset_indices = self._select_coreset_grand()
            elif baseline == "influential":
                coreset_indices = self._select_coreset_influential()
            elif baseline == "moderate":
                coreset_indices = self._select_coreset_moderate()
            elif baseline == "ccs":
                coreset_indices = self._select_coreset_ccs()
            elif baseline == "probabilistic":
                coreset_indices = self._select_coreset_probabilistic()
            else:
                raise ValueError(f"Unknown baseline: {baseline}")
            
            # Train and evaluate
            accuracy, coreset_size = self._train_and_evaluate(coreset_indices, baseline)
            
            # Store results
            self.results[baseline] = {
                "accuracy": accuracy,
                "coreset_size": coreset_size,
                "improvement_over_threshold": accuracy - self.performance_threshold,
                "coreset_ratio": coreset_size / len(self.train_dataset)
            }
            
            logger.info(f"{baseline}: Accuracy={accuracy:.2f}%, Coreset size={coreset_size}")
        
        # Save results
        results_file = os.path.join(self.save_dir, "baseline_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("BASELINE COMPARISON RESULTS")
        print("="*80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Full data accuracy: {self.full_data_accuracy:.2f}%")
        print(f"Performance threshold: {self.performance_threshold:.2f}%")
        print(f"Target coreset size: {self.coreset_size}")
        print("-" * 80)
        
        for baseline in self.baselines:
            result = self.results[baseline]
            print(f"{baseline:15}: {result['accuracy']:6.2f}% ({result['coreset_size']:5d} samples) "
                  f"({result['improvement_over_threshold']:6.2f}% over threshold)")
        
        print("="*80)
        
        return self.results

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Comparison for Coreset Selection')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn'],
                       help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('--noise_rate', type=float, default=0.3,
                       help='Noise rate for label corruption (default: 0.3)')
    parser.add_argument('--coreset_size', type=int, default=1000,
                       help='Maximum coreset size (default: 1000)')
    parser.add_argument('--tolerance', type=str, default='15%',
                       help='Performance tolerance (e.g., 15% or 2.0) (default: 15%)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    comparator = BaselineComparison(args)
    results = comparator.run_comparison()