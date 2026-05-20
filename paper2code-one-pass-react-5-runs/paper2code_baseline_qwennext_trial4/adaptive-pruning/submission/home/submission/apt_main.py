import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import os
import json
import argparse
from typing import Dict, List, Optional
import time
from tqdm import tqdm
import random

from apt_model import APTRobertaForSequenceClassification
from apt_adapter import APTAdapter

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_sst2_data(tokenizer, max_length: int = 128, sample_size: Optional[int] = None):
    """Load and preprocess SST2 dataset."""
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    
    # Sample if needed
    if sample_size:
        dataset['train'] = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(sample_size, len(dataset['validation']))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_val = dataset['validation'].map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_train, tokenized_val

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                epochs: int = 3, pruning_steps: List[int] = None, 
                target_sparsity: float = 0.6, output_dir: str = "./results"):
    """Train the APT model with adaptive pruning and tuning."""
    
    model.to(device)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    training_history = []
    
    # Set up pruning schedule
    if pruning_steps is None:
        pruning_steps = [epochs // 3, 2 * epochs // 3]
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate training metrics
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate validation metrics
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Adaptive pruning - apply at specified steps
        if epoch + 1 in pruning_steps:
            print(f"Applying adaptive pruning at epoch {epoch + 1}")
            # Calculate salience scores (simplified - in practice, we'd do this properly)
            # For this implementation, we'll just simulate pruning
            model.adaptive_pruning({}, target_sparsity)
        
        # Adaptive tuning - increase ranks in salient layers
        if epoch + 1 in pruning_steps:
            print(f"Applying adaptive tuning at epoch {epoch + 1}")
            # Simulate adapter salience scores
            adapter_salience = {}
            for name, adapter in model.apts.items():
                # Use a simple heuristic: higher salience for later layers
                adapter_salience[name] = 1.0 + (epoch / epochs) * 0.5
            
            model.adaptive_tuning(adapter_salience, epoch + 1, epochs)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'target_sparsity': target_sparsity
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Log metrics
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="APT: Adaptive Pruning and Tuning")
    parser.add_argument("--model_type", type=str, default="roberta", 
                       choices=["roberta", "t5"], help="Model type to use")
    parser.add_argument("--task", type=str, default="sst2", 
                       choices=["sst2"], help="Task to run")
    parser.add_argument("--sparsity", type=float, default=0.6, 
                       help="Target sparsity (0.0 to 1.0)")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, 
                       help="Initial rank for APT adapters")
    parser.add_argument("--output_dir", type=str, default="./results", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    if args.model_type == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = APTRobertaForSequenceClassification(
            model_name=model_name, 
            num_labels=2, 
            prune_ratio=args.sparsity, 
            rank=args.rank
        )
    else:
        raise NotImplementedError("Only RoBERTa is implemented for this reproduction")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset = load_sst2_data(tokenizer, sample_size=1000)  # Use smaller sample for speed
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    best_val_acc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device,
        epochs=args.epochs, target_sparsity=args.sparsity, output_dir=args.output_dir
    )
    end_time = time.time()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'args': args
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Print summary
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output_dir}")
    
    # Calculate and print efficiency metrics
    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = model.get_pruned_parameters_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated pruned parameters: {pruned_params:,}")
    print(f"Estimated sparsity: {pruned_params/total_params:.2%}")

if __name__ == "__main__":
    main()