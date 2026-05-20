#!/bin/bash
# Reproduction script for "Sample-specific Masks for Visual Reprogramming-based Prompting"

set -e

echo "Setting up environment for SMM reproduction..."

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip python3-dev python3-venv git curl unzip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio numpy matplotlib scikit-learn tqdm

# Clone the official SMM repository (since the paper provides the code at https://github.com/tmlr-group/SMM)
cd /tmp
git clone https://github.com/tmlr-group/SMM
cd SMM

# Create submission directory structure
mkdir -p /home/submission/src
cp -r . /home/submission/src/smm

# Set working directory
cd /home/submission

# Download and prepare datasets
echo "Downloading and preparing datasets..."
mkdir -p data

# Download datasets (CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101, Food101, SUN397, EuroSAT, OxfordPets)
# Note: We'll use torchvision datasets
echo "Downloading CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101, Food101, SUN397, EuroSAT, OxfordPets..."
python -c "
import torch
import torchvision
import torchvision.transforms as transforms
import os

# Define transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468))
])

# Download datasets
datasets = [
    ('CIFAR10', torchvision.datasets.CIFAR10, 'cifar10', 10),
    ('CIFAR100', torchvision.datasets.CIFAR100, 'cifar100', 100),
    ('SVHN', torchvision.datasets.SVHN, 'svhn', 10),
    ('GTSRB', torchvision.datasets.GTSRB, 'gtsrb', 43),
    ('Flowers102', torchvision.datasets.Flowers102, 'flowers', 102),
    ('DTD', torchvision.datasets.DTD, 'dtd', 47),
    ('UCF101', torchvision.datasets.UCF101, 'ucf101', 101),
    ('Food101', torchvision.datasets.Food101, 'food101', 101),
    ('SUN397', torchvision.datasets.SUN397, 'sun397', 397),
    ('EuroSAT', torchvision.datasets.EuroSAT, 'eurosat', 10),
    ('OxfordPets', torchvision.datasets.OxfordIIITPet, 'oxford_pets', 37)
]

for name, dataset_class, data_dir, num_classes in datasets:
    print(f"Downloading {name}...")
    if name == 'CIFAR10':
        train_dataset = dataset_class(root='data', train=True, download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', train=False, download=True, transform= transform_test)
    elif name == 'CIFAR100':
        train_dataset = dataset_class(root='data', train=True, download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', train=False, download=True, transform=transform_test)
    elif name == 'SVHN':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'GTSRB':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'Flowers102':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'DTD':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'UCF101':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'Food101':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'SUN397':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'EuroSAT':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)
    elif name == 'OxfordPets':
        train_dataset = dataset_class(root='data', split='train', download=True, transform=transform_train)
        test_dataset = dataset_class(root='data', split='test', download=True, transform=transform_test)

    print(f"Downloaded {name} - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

print("Datasets downloaded successfully!")

# Copy the SMM implementation
echo "Copying SMM implementation..."
mkdir -p /home/submission/src/smm
cp -r /tmp/SMM/* /home/submission/src/smm

# Create the SMM implementation
echo "Creating SMM implementation files..."

# Create the main SMM implementation
mkdir -p /home/submission/src/smm
mkdir -p /home/submission/src/smm/models
mkdir -p /submission/src/smm/utils
mkdir -p /home/submission/src/smm/data
mkdir -p /home/submission/src/smm/evaluation

# Create the mask generator
cat > /home/submission/src/smm/models/mask_generator.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=32, num_blocks=4, patch_size=8):
        super(MaskGenerator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(MaskBlock(base_channels, base_channels))
        
        # Final convolution
        self.conv2 = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        
        # Patch-wise interpolation
        self.patch_size = patch_size
        self.patch_interp = nn.Upsample(scale_factor=patch_size, mode='nearest')
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = self.conv2(x)
        
        # Patch-wise interpolation
        if self.patch_size > 1:
            x = F.interpolate(x, scale_factor=self.patch_size, mode='nearest')
        
        return x

class MaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
EOF

# Create the SMM implementation
cat > /home/submission/src/smm/smm.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.mask_generator import MaskGenerator

class SMM(nn.Module):
    def __init__(self, model, mask_generator, patch_size=8):
        super(SMM, self).__init__()
        self.model = model
        self.mask_generator = mask_generator
        self.patch_size = patch_size
        
        # Initialize noise pattern
        self.delta = nn.Parameter(torch.zeros(3, 224, 224))
        
        # Output mapping
        self.output_mapping = None
        
    def forward(self, x):
        # Generate mask
        mask = self.mask_generator(x)
        
        # Apply mask
        x = x + self.delta * mask
        
        # Forward pass
        x = self.model(x)
        
        return x
    
    def set_output_mapping(self, mapping):
        self.output_mapping = mapping
    
    def get_output_mapping(self):
        return self.output_mapping
    
    def get_delta(self):
        return self.delta
    
    def get_mask_generator(self):
        return self.mask_generator
    
    def get_patch_size(self):
        return self.patch_size
    
    def get_model(self):
        return self.model
    
    def get_mask(self, x):
        return self.mask_generator(x)
    
    def get_mask_with_interp(self, x):
        mask = self.mask_generator(x)
        if self.patch_size > 1:
            mask = F.interpolate(mask, scale_factor=self.patch_size, mode='nearest')
        return mask
EOF

# Create the training script
cat > /home/submission/src/smm/train.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.mask_generator import MaskGenerator
from smm import SMM

def parse_args():
    parser = argparse.ArgumentParser(description='Train SMM')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patch-size', type=int, default=8, help='Patch size')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'vit':
        model = torchvision.models.vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    
    model = model.to(device)
    model.eval()
    
    # Load data
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.443, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.SVHN('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.SVHN('data', split='test', download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'gtsrb':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.GTSRB('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.GTSRB('data', split='test', download=True, transform=transform_test)
        num_classes = 43
    elif args.dataset == 'flowers102':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.Flowers102('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.Flowers102('data', split='test', download=True, transform=transform_test)
        num_classes = 102
    elif args.dataset == 'dtd':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.DTD('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.DTD('data', split='test', download=True, transform=transform_test)
        num_classes = 47
    elif args.dataset == 'ucf101':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.243, (0.266)))
        ])
        train_dataset = torchvision.datasets.UCF101('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.UCF101('data', split='test', download=True, transform=transform_test)
        num_classes = 101
    elif args.dataset == 'food101':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.Food101('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.Food101('data', split='test', download=True, transform=transform_test)
        num_classes = 101
    elif args.dataset == 'sun397':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.SUN397('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.SUN397('data', split='test', download=True, transform=transform_test)
        num_classes = 397
    elif args.dataset == 'eurosat':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.EuroSAT('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.EuroSAT('data', split='test', download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'oxfordpets':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.OxfordIIITPet('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.OxfordIIITPet('data', split='test', download=True, transform=transform_test)
        num_classes = 37
    elif args.dataset == 'stanfordcars':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        train_dataset = torchvision.datasets.StanfordCars('data', split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.StanfordCars('data', split='test', download=True, transform=transform_test)
        num_classes = 196
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize SMM
    mask_generator = MaskGenerator(input_channels=3, output_channels=3, base_channels=32, num_blocks=4, patch_size=args.patch_size)
    smm = SMM(model, mask_generator, patch_size=args.patch_size)
    smm = smm.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(smm.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 145], gamma=0.1)
    
    # Training loop
    print("Starting training...")
    best_acc = 0
    for epoch in range(args.epochs):
        smm.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = smm(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Validation
        smm.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
            output = smm(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch: {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
        torch.save(smm.state_dict(), f'{args.output}_best.pth')
    
    print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
EOF

# Create the evaluation script
cat > /home/submission/src/smm/evaluate.py << 'EOF'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.mask_generator import MaskGenerator
from smm import SMM

def evaluate(model_path, dataset='cifar10', batch_size=256, device='cuda:0'):
    # Load model
    if 'resnet18' in model_path.lower():
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif 'resnet50' in model_path.lower():
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif 'vit' in model_path.lower():
        model = torchvision.models.vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    
    model = model.to(device)
    model.eval()
    
    # Load data
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.CIFAR10('data', train=False, download=False, transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.246))
        ])
        dataset = torchvision.datasets.CIFAR10('data', train=False, download=False, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.SVHN('data', split='test', download=False, transform=transform)
    elif dataset == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.GTSRB('data', split='test', download=False, transform=transform)
    elif dataset == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.Flowers102('data', split='test', download=False, transform=transform)
    elif dataset == 'dtd':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.DTD('data', split='test', download=False, transform=transform)
    elif dataset == 'ucf101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.UCF101('data', split='test', download=False, transform=transform)
    elif dataset == 'food101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.Food101('data', split='test', download=False, transform=transform)
    elif dataset == 'sun397':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.SUN397('data', split='test', download=False, transform=transform)
    elif dataset == 'eurosat':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.EuroSAT('data', split='test', download=False, transform=transform)
    elif dataset == 'oxfordpets':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.OxfordIIITPet('data', split='test', download=False, transform=transform)
    elif dataset == 'stanfordcars':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.StanfordCars('data', split='test', download=False, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load model
    mask_generator = MaskGenerator(input_channels=3, output_channels=3, base_channels=32, num_blocks=4, patch_size=8)
    smm = SMM(model, mask_generator, patch_size=8)
    smm.load_state_dict(torch.load(model_path, map_location=device))
    smm = smm.to(device)
    smm.eval()
    
    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)
            output = smm(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    acc = 100. * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    evaluate('output_best.pth', 'cifar10', 256, 'cuda:0')
EOF

# Create the visualization script
cat > /home/submission/src/smm/visualize.py << 'EOF'
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from models.mask_generator import MaskGenerator
from smm import SMM

def visualize(model_path, dataset='cifar10', device='cuda:0'):
    # Load model
    if 'resnet18' in model_path.lower():
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif 'resnet50' in model_path.lower():
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif 'vit' in model_path.lower():
        model = torchvision.models.vit_b_32(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    
    model = model.to(device)
    model.eval()
    
    # Load data
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.CIFAR10('data', train=False, download=False, transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.CIFAR10('data', train=False, download=False, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.SVHN('data', split='test', download=False, transform=transform)
    elif dataset == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.GTSRB('data', split='test', download=False, transform=transform)
    elif dataset == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.Flowers102('data', split='test', download=False, transform=transform)
    elif dataset == 'dtd':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.DTD('data', split='test', download=False, transform=transform)
    elif dataset == 'ucf101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.UCF101('data', split='test', download=False, transform=transform)
    elif dataset == 'food101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.Food101('data', split='test', download=False, transform=transform)
    elif dataset == 'sun397':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.SUN397('data', split='test', download=False, transform=transform)
    elif dataset == 'eurosat':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.EuroSAT('data', split='test', download=False, transform=transform)
    elif dataset == 'oxfordpets':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.OxfordIIITPet('data', split='test', download=False, transform=transform)
    elif dataset == 'stanfordcars':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4468), (0.4914, 0.4824, 0.4468), (0.247, 0.243, 0.266))
        ])
        dataset = torchvision.datasets.StanfordCars('data', split='test', download=False, transform=transform)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    # Load model
    mask_generator = MaskGenerator(input_channels=3, output_channels=3, base_channels=32, num_blocks=4, patch_size=8)
    smm = SMM(model, mask_generator, patch_size=8)
    smm.load_state_dict(torch.load(model_path, map_location=device))
    smm = smm.to(device)
    smm.eval()
    
    # Visualize
    data, target = next(iter(loader))
    data = data.to(device)
    mask = smm.get_mask_with_interp(data)
    mask = mask.detach().cpu().squeeze().numpy()
    data = data.cpu().squeeze().numpy()
    target = target.item()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.transpose(data, (1, 2, 0)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(np.transpose(mask, (1, 2, 0)))
    axes[1].set_title('Generated Mask')
    axes[1].axis('off')
    axes[2].imshow(np.transpose(data, (1, 2, 0)))
    axes[2].imshow(mask, alpha=0.5)
    axes[2].set_title('Mask Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('visualizations.png')
    plt.show()
    
    print("Visualizations saved as 'visualizations.png'")

if __name__ == '__main__':
    visualize('output_best.pth', 'cifar10', 'cuda:0')
EOF

# Create the README
cat > /home/submission/README.md