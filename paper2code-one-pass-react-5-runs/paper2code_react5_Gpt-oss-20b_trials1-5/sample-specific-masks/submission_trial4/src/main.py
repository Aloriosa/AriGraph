import torch
import torch.optim as optim
from tqdm import tqdm

from .data import get_dataloaders
from .model import BaseModel, SMMModel
from .utils import set_seed
from .mapping import random_mapping, compute_mapping
from .pretrained_backbone import PretrainedBackbone
from .config import *

def get_optimizer(model, lr):
    """Collect all trainable parameters of the model."""
    params = list(model.delta.parameters())
    if hasattr(model, "mask_gen"):
        params += list(model.mask_gen.parameters())
    return optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)

def lr_scheduler(optimizer, epoch):
    if epoch in LR_SCHEDULE:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= LR_GAMMA

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc=f"Train epoch {epoch+1}", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = model.loss(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            selected_logits = logits[:, model.mapping]
            preds = selected_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Load data
    train_loader, test_loader = get_dataloaders(DATASET, batch_size=BATCH_SIZE)

    # Determine number of target classes
    if DATASET == "cifar10":
        num_target = 10
    elif DATASET == "cifar100":
        num_target = 100
    elif DATASET == "svhn":
        num_target = 10
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")

    # Create mapping tensor
    if MAPPING_STRATEGY == "random":
        mapping_tensor = random_mapping(num_target, NUM_SOURCE_CLASSES)
    elif MAPPING_STRATEGY == "frequent":
        # Compute mapping once using the frozen ImageNet backbone
        pretrained_model = PretrainedBackbone(BACKBONE).to(device)
        mapping_tensor = compute_mapping(train_loader, pretrained_model,
                                         num_target, NUM_SOURCE_CLASSES,
                                         device=device)
    elif MAPPING_STRATEGY == "iterative":
        # Initial mapping is random; will be updated each epoch
        mapping_tensor = random_mapping(num_target, NUM_SOURCE_CLASSES)
    else:
        raise ValueError(f"Unsupported mapping strategy: {MAPPING_STRATEGY}")

    # Instantiate models
    baseline = BaseModel(backbone_name=BACKBONE,
                         num_target_classes=num_target,
                         mask_size=BASE_MASK_SIZE,
                         mapping_tensor=mapping_tensor,
                         device=device).to(device)

    smm      = SMMModel(backbone_name=BACKBONE,
                        num_target_classes=num_target,
                        mask_size=BASE_MASK_SIZE,
                        mapping_tensor=mapping_tensor,
                        device=device).to(device)

    # Optimizers
    opt_baseline = get_optimizer(baseline, BASE_LR)
    opt_smm      = get_optimizer(smm, BASE_LR)

    # Training loop
    for epoch in range(EPOCHS):
        lr_scheduler(opt_baseline, epoch)
        lr_scheduler(opt_smm, epoch)

        loss_b = train_one_epoch(baseline, train_loader, opt_baseline, device, epoch)
        loss_s = train_one_epoch(smm, train_loader, opt_smm, device, epoch)

        acc_b = evaluate(baseline, test_loader, device)
        acc_s = evaluate(smm, test_loader, device)

        print(f"Epoch {epoch+1:3d} | Baseline loss: {loss_b:.4f} | SMM loss: {loss_s:.4f}")
        print(f"           | Baseline acc : {acc_b:.2f}% | SMM acc : {acc_s:.2f}%")

        # Update mapping if iterative strategy is used
        if MAPPING_STRATEGY == "iterative":
            new_mapping = compute_mapping(train_loader, smm, num_target,
                                          NUM_SOURCE_CLASSES, device=device)
            smm.mapping = new_mapping
            baseline.mapping = new_mapping  # keep same mapping for fairness

    # Final evaluation
    final_acc_b = evaluate(baseline, test_loader, device)
    final_acc_s = evaluate(smm, test_loader, device)

    print("\n===== Final results =====")
    print(f"Baseline accuracy: {final_acc_b:.2f}%")
    print(f"SMM accuracy:     {final_acc_s:.2f}%")

    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Baseline accuracy: {final_acc_b:.2f}%\n")
        f.write(f"SMM accuracy: {final_acc_s:.2f}%\n")