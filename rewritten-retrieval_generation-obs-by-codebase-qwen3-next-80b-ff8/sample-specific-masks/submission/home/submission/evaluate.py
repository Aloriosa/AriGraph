import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import the model components from the linked code assets
from instance_model import AttributeNet, InstancewiseVisualPrompt

def get_dataset(dataset_name, image_size=224):
    """Load and preprocess the dataset"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'oxford_pets':
        dataset = torchvision.datasets.OxfordIIITPet(
            root='./data', split='test', download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset

def get_pretrained_model(model_name, num_classes):
    """Load a pre-trained model and freeze its parameters"""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'vit_b_32':
        model = torchvision.models.vit_b_32(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier layer
    if model_name.startswith('resnet'):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'vit_b_32':
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    
    return model

def evaluate(model, visual_prompt, dataloader, device):
    """Evaluate the model"""
    model.eval()
    visual_prompt.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * len(dataloader.dataset.classes)
    class_total = [0] * len(dataloader.dataset.classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply visual reprogramming
            reprogrammed_images = visual_prompt(images)
            
            # Forward pass through frozen model
            outputs = model(reprogrammed_images)
            
            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Class-wise accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate overall accuracy
    overall_acc = 100. * correct / total
    
    # Calculate class-wise accuracy
    class_acc = []
    for i in range(len(dataloader.dataset.classes)):
        if class_total[i] > 0:
            class_acc.append(100. * class_correct[i] / class_total[i])
        else:
            class_acc.append(0)
    
    return overall_acc, class_acc

def main():
    parser = argparse.ArgumentParser(description='Evaluate Sample-Specific Masking for Visual Reprogramming')
    parser.add_argument('--dataset', type=str, default='oxford_pets', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='Pre-trained model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for mask generation')
    parser.add_argument('--layers', type=int, default=5, help='Number of layers in mask generator')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in mask (3 for RGB)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, default='./evaluation_results.txt', help='Output file for results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(args.dataset, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get number of classes
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Load pre-trained model
    print(f"Loading pre-trained {args.model}...")
    pretrained_model = get_pretrained_model(args.model, num_classes)
    pretrained_model = pretrained_model.to(device)
    
    # Create visual prompting module
    print("Creating visual prompting module...")
    visual_prompt = InstancewiseVisualPrompt(
        size=args.image_size,
        layers=args.layers,
        patch_size=args.patch_size,
        channels=args.channels
    )
    visual_prompt = visual_prompt.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    visual_prompt.load_state_dict(checkpoint['visual_prompt_state_dict'])
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("Evaluating model...")
    overall_acc, class_acc = evaluate(pretrained_model, visual_prompt, dataloader, device)
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n")
        f.write(f"Class-wise Accuracy:\n")
        for i, acc in enumerate(class_acc):
            f.write(f"  Class {i}: {acc:.2f}%\n")
        f.write(f"\nDetailed Results:\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Image Size: {args.image_size}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Channels: {args.channels}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
    
    print(f"Results saved to {args.output_file}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")

if __name__ == '__main__':
    main()