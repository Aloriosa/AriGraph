import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from flaml import tune
from flaml.tune.searcher.flow2 import FLOW2
import time
import logging
from tqdm import tqdm
from models import ConvNetCIFAR, ResNet, BasicBlock
from datasets_utils.cifar10 import CIFAR10
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LBCS:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = args.dataset
        self.noise_rate = args.noise_rate
        self.coreset_size = args.coreset_size
        self.tolerance = args.tolerance
        self.train_epochs = args.train_epochs
        self.save_dir = args.save_dir
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Initialize datasets
        self.train_dataset, self.test_dataset, self.val_dataset = self._load_datasets()
        
        # Get full dataset performance baseline
        self.full_data_accuracy = self._compute_full_data_baseline()
        
        # Define performance constraint (must not degrade below baseline)
        if '%' in self.tolerance:
            tolerance_percent = float(self.tolerance.replace('%', ''))
            self.performance_threshold = self.full_data_accuracy * (1 - tolerance_percent / 100)
        else:
            self.performance_threshold = self.full_data_accuracy - float(self.tolerance)
        
        logger.info(f"Full data baseline accuracy: {self.full_data_accuracy:.2f}%")
        logger.info(f"Performance threshold: {self.performance_threshold:.2f}%")
        
        # Define model architecture based on dataset
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Define optimizer for training
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Define search space for coreset selection
        self.search_space = {
            "coreset_size": tune.randint(1, self.coreset_size),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128])
        }
        
        # Define lexicographic objectives for FLOW2 searcher
        self.lexico_objectives = {
            "metrics": ["accuracy", "coreset_size"],
            "modes": ["max", "min"],
            "tolerances": {"accuracy": self.tolerance, "coreset_size": 0},
            "targets": {"accuracy": self.full_data_accuracy, "coreset_size": 0}
        }
        
        # Initialize FLOW2 searcher with lexicographic optimization
        self.searcher = FLOW2(
            init_config={"coreset_size": self.coreset_size, "learning_rate": 0.1, "batch_size": 128},
            metric="accuracy",
            mode="max",
            space=self.search_space,
            lexico_objectives=self.lexico_objectives,
            seed=42
        )
        
        # Store coreset indices
        self.coreset_indices = None
        self.best_config = None
        
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
        for epoch in range(10):  # Train for 10 epochs to get good baseline
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
    
    def _train_on_coreset(self, coreset_indices, config):
        """Train model on selected coreset and evaluate on validation set"""
        # Create coreset dataset
        coreset_dataset = Subset(self.train_dataset, coreset_indices)
        coreset_loader = DataLoader(coreset_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model.train()
        for epoch in range(self.train_epochs):
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
                logger.info(f"Epoch {epoch+1}/{self.train_epochs}, Loss: {total_loss/len(coreset_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")
        
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
    
    def _compute_importance_scores(self):
        """Compute importance scores for all samples using gradient norms"""
        logger.info("Computing importance scores for all samples...")
        
        # Create data loader for full training data
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        # Initialize model for gradient computation
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        importance_scores = []
        
        for batch_idx, (data, target, indices) in enumerate(tqdm(train_loader, desc="Computing importance scores")):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute gradient norms for each sample
            for param in model.parameters():
                if param.grad is not None:
                    # Compute norm of gradients for each sample
                    grad_norm = torch.norm(param.grad.view(param.grad.size(0), -1), dim=1)
                    importance_scores.extend(grad_norm.cpu().numpy())
        
        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
        
        return importance_scores
    
    def _select_coreset(self):
        """Select coreset using lexicographic bilevel optimization"""
        logger.info("Starting coreset selection with LBCS...")
        
        # Compute importance scores for all samples
        importance_scores = self._compute_importance_scores()
        
        # Sort samples by importance score (descending)
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Start with largest possible coreset and iteratively reduce
        current_coreset_size = min(self.coreset_size, len(self.train_dataset))
        best_coreset_indices = sorted_indices[:current_coreset_size]
        best_accuracy = 0.0
        
        # Use FLOW2 searcher to find optimal coreset size
        def objective(config):
            # Use the coreset size from config
            coreset_size = config["coreset_size"]
            # Select top k samples by importance score
            coreset_indices = sorted_indices[:coreset_size]
            
            # Train on coreset and evaluate
            accuracy, _ = self._train_on_coreset(coreset_indices, config)
            
            # Return results for lexicographic optimization
            return {
                "accuracy": accuracy,
                "coreset_size": coreset_size,
                "importance_score": np.mean(importance_scores[coreset_indices])
            }
        
        # Run FLOW2 search
        analysis = tune.run(
            objective,
            search_alg=self.searcher,
            num_samples=20,  # Number of configurations to try
            config=self.search_space,
            metric="accuracy",
            mode="max",
            verbose=1
        )
        
        # Get best configuration
        best_config = analysis.get_best_config(metric="accuracy", mode="max")
        best_coreset_size = best_config["coreset_size"]
        best_coreset_indices = sorted_indices[:best_coreset_size]
        
        logger.info(f"Best coreset size: {best_coreset_size}")
        logger.info(f"Best accuracy: {analysis.best_trial.last_result['accuracy']:.2f}%")
        
        self.best_config = best_config
        self.coreset_indices = best_coreset_indices
        return best_coreset_indices
    
    def run(self):
        """Run the complete LBCS algorithm"""
        logger.info("Starting LBCS algorithm...")
        
        # Select coreset
        self.coreset_indices = self._select_coreset()
        
        # Train final model on selected coreset
        logger.info("Training final model on selected coreset...")
        final_accuracy = self._train_final_model()
        
        # Evaluate on test set
        test_accuracy = self._evaluate_on_test()
        
        # Save results
        results = {
            "dataset": self.dataset_name,
            "full_data_accuracy": self.full_data_accuracy,
            "performance_threshold": self.performance_threshold,
            "selected_coreset_size": len(self.coreset_indices),
            "final_accuracy": final_accuracy,
            "test_accuracy": test_accuracy,
            "best_config": self.best_config,
            "coreset_indices": self.coreset_indices.tolist() if self.coreset_indices is not None else []
        }
        
        # Save results to file
        results_file = os.path.join(self.save_dir, "results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("LBCS REPRODUCTION RESULTS")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Full data accuracy: {self.full_data_accuracy:.2f}%")
        print(f"Performance threshold: {self.performance_threshold:.2f}%")
        print(f"Selected coreset size: {len(self.coreset_indices)}")
        print(f"Final accuracy on validation set: {final_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print(f"Accuracy improvement over threshold: {final_accuracy - self.performance_threshold:.2f}%")
        print(f"Core set size reduction: {100 * (1 - len(self.coreset_indices)/len(self.train_dataset)):.1f}%")
        print("="*60)
        
        return results
    
    def _train_final_model(self):
        """Train final model on selected coreset"""
        coreset_dataset = Subset(self.train_dataset, self.coreset_indices)
        coreset_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=4)
        
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.best_config["learning_rate"], momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(self.train_epochs):
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
                logger.info(f"Final Epoch {epoch+1}/{self.train_epochs}, Loss: {total_loss/len(coreset_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")
        
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
        return accuracy
    
    def _evaluate_on_test(self):
        """Evaluate final model on test set"""
        coreset_dataset = Subset(self.train_dataset, self.coreset_indices)
        model = self._create_model().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.best_config["learning_rate"], momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train on coreset
        coreset_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True, num_workers=4)
        model.train()
        for epoch in range(self.train_epochs):
            total_loss = 0
            for batch_idx, (data, target, _) in enumerate(coreset_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target, _ in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Lexicographic Bilevel Coreset Selection')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'svhn'],
                       help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('--noise_rate', type=float, default=0.3,
                       help='Noise rate for label corruption (default: 0.3)')
    parser.add_argument('--coreset_size', type=int, default=1000,
                       help='Maximum coreset size (default: 1000)')
    parser.add_argument('--tolerance', type=str, default='15%',
                       help='Performance tolerance (e.g., 15% or 2.0) (default: 15%)')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    lbcs = LBCS(args)
    results = lbcs.run()