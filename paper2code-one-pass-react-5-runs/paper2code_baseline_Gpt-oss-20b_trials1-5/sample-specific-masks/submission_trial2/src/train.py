import argparse
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import set_seed, get_random_mapping, train_one_epoch, evaluate
from model import SMMReprogramming

def get_dataloaders(batch_size=256, num_workers=4):
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return trainloader, testloader

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                num_workers=args.workers)

    # Instantiate model
    model = SMMReprogramming(pretrained_backbone='resnet18',
                             target_classes=10,
                             patch_size=2).to(device)

    # Random output mapping from pretrained 1000 classes to 10 target classes
    mapping = get_random_mapping(num_pretrained=1000, num_target=10, seed=args.seed)
    model.set_mapping(mapping)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.delta, 'lr': args.lr_delta},
                           {'params': model.mask_gen.parameters(), 'lr': args.lr_mask}],
                          lr=args.lr_base)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/smm_epoch{epoch}.pth')
        print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')

    print(f'Best Test Accuracy: {best_acc:.4f}')
    # Save final results
    with open('output.txt', 'w') as f:
        f.write(f'Best Test Accuracy: {best_acc:.4f}\\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMM Visual Reprogramming on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='num workers')
    parser.add_argument('--lr-base', type=float, default=1e-3, help='base learning rate for mask')
    parser.add_argument('--lr-mask', type=float, default=1e-3, help='learning rate for mask generator')
    parser.add_argument('--lr-delta', type=float, default=1e-3, help='learning rate for delta')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    main(args)