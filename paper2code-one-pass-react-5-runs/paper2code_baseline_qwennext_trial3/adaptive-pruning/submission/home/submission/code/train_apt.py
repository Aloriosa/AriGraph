#!/usr/bin/env python3
"""
Main training script for APT - Adaptive Pruning and Tuning
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
import logging
from tqdm import tqdm
import json
import time
from typing import Dict, List, Optional, Tuple

# Import APT components
from apt_adapter import APTAdapter
from salience_calculator import SalienceCalculator
from pruning_scheduler import PruningScheduler
from distillation import SelfDistillation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APTTrainer:
    """
    Main APT trainer class
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        self.model_name = self._get_model_name()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._load_model()
        
        # Initialize APT components
        self.salience_calculator = SalienceCalculator(self.model)
        self.pruning_scheduler = PruningScheduler(
            self.model, 
            target_sparsity=args.sparsity,
            initial_sparsity=0.0,
            total_steps=args.epochs * 100,  # Estimate steps
            schedule_type='cubic'
        )
        self.distillation = SelfDistillation(
            self.model,
            distill_ratio=0.5,
            distill_start_step=0,
            distill_end_step=args.epochs * 50
        )
        
        # Initialize APT adapters
        self.apts = {}
        self._add_apts_to_model()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Move model to device
        self.model.to(self.device)
        
    def _get_model_name(self) -> str:
        """Get model name based on args"""
        model_map = {
            'roberta': 'roberta-base',
            't5': 't5-base',
            'llama': 'gpt2'  # Using GPT-2 as proxy for LLaMA due to size constraints
        }
        return model_map.get(self.args.model_type, 'roberta-base')
        
    def _load_model(self):
        """Load appropriate model based on task"""
        if self.args.model_type == 'roberta':
            if self.args.task == 'sst2':
                return AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=2
                )
            elif self.args.task == 'mnli':
                return AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=3
                )
        elif self.args.model_type == 't5':
            # For T5, we'll use a sequence classification wrapper
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=3 if self.args.task == 'mnli' else 2
            )
        elif self.args.model_type == 'llama':
            # For LLaMA, we'll use a causal language model
            return AutoModelForCausalLM.from_pretrained(self.model_name)
            
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        )
        
    def _add_apts_to_model(self):
        """Add APT adapters to model layers"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                # Add APT adapters to attention layers
                if 'attention' in name.lower() or 'self_attn' in name.lower():
                    if hasattr(module, 'query') and hasattr(module, 'value'):
                        # Add APT adapter for query and value projections
                        self._add_apts_to_linear_layer(module.query, f"{name}.query")
                        self._add_apts_to_linear_layer(module.value, f"{name}.value")
                        
                # Add APT adapters to FFN layers
                elif 'feedforward' in name.lower() or 'ffn' in name.lower():
                    if hasattr(module, 'dense') and hasattr(module, 'intermediate'):
                        self._add_apts_to_linear_layer(module.dense, f"{name}.dense")
                        self._add_apts_to_linear_layer(module.intermediate, f"{name}.intermediate")
                        
    def _add_apts_to_linear_layer(self, linear_layer: nn.Linear, layer_name: str):
        """Add APT adapter to a linear layer"""
        input_dim = linear_layer.in_features
        output_dim = linear_layer.out_features
        
        # Create APT adapter
        apt = APTAdapter(input_dim, output_dim, rank=8, use_pruning=True)
        apt.to(self.device)
        
        # Store adapter
        self.apts[layer_name] = apt
        
        # Replace forward pass
        original_forward = linear_layer.forward
        
        def new_forward(x):
            # Apply original linear transformation
            output = original_forward(x)
            
            # Apply APT adapter
            apt_output = apt(x, frozen_weight=None)
            
            # Combine outputs
            return output + apt_output
            
        linear_layer.forward = new_forward
        
    def load_data(self):
        """Load and preprocess dataset"""
        data_path = f'data/{self.args.task}'
        
        if self.args.task in ['sst2', 'mnli']:
            dataset = load_from_disk(data_path)
            
            def tokenize_function(examples):
                if self.args.task == 'sst2':
                    return self.tokenizer(
                        examples['sentence'],
                        padding='max_length',
                        truncation=True,
                        max_length=128
                    )
                else:  # mnli
                    return self.tokenizer(
                        examples['premise'],
                        examples['hypothesis'],
                        padding='max_length',
                        truncation=True,
                        max_length=128
                    )
                    
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Set format for PyTorch
            tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            # Create dataloaders
            train_loader = DataLoader(tokenized_dataset['train'], batch_size=self.args.batch_size, shuffle=True)
            val_loader = DataLoader(tokenized_dataset['validation'], batch_size=self.args.batch_size)
            
            return train_loader, val_loader
            
        elif self.args.task == 'alpaca':
            # Load simulated Alpaca dataset
            with open('data/alpaca.json', 'r') as f:
                data = json.load(f)
                
            # Tokenize text
            def tokenize_function(texts):
                return self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
            # Create dummy labels (we'll use next token prediction)
            inputs = tokenize_function(data['text'])
            dataset = torch.utils.data.TensorDataset(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
            
            train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            val_loader = None  # No validation for Alpaca
            
            return train_loader, val_loader
            
        return None, None
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if self.args.task == 'alpaca':
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = input_ids.clone()
            else:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.args.task == 'alpaca':
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            # Compute distillation loss if applicable
            distill_loss = torch.tensor(0.0)
            if self.args.model_type == 'roberta' and epoch > 0:
                # For distillation, we need hidden states
                # This is a simplified version
                with torch.no_grad():
                    teacher_outputs = self.distillation.teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    
                # In practice, we'd need to capture hidden states
                # This is a placeholder
                distill_loss = torch.tensor(0.0)
                
            # Apply distillation weight
            distill_weight = self.distillation.get_distill_weight(epoch * len(train_loader) + batch_idx)
            total_loss_value = loss + distill_weight * distill_loss
            
            # Backward pass
            total_loss_value.backward()
            
            # Update APT adapters
            self._update_apts()
            
            # Update optimizer
            self.optimizer.step()
            
            total_loss += total_loss_value.item()
            num_batches += 1
            
            # Update pruning scheduler
            if batch_idx % 10 == 0:
                self._update_pruning_scheduler()
                
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
        return total_loss / num_batches
        
    def _update_apts(self):
        """Update APT adapters with new weights"""
        # This is a placeholder - in practice we'd update based on gradients
        pass
        
    def _update_pruning_scheduler(self):
        """Update pruning scheduler and apply masks"""
        # Get current step
        current_step = self.pruning_scheduler.step
        
        # Get salience scores (simplified - in practice we'd compute from activations)
        # This is a placeholder implementation
        salience_scores = {}
        for name in self.apts.keys():
            # Create dummy salience scores
            if 'head' in name:
                salience_scores[name] = torch.rand(self.apts[name].input_dim)
            elif 'neuron' in name:
                salience_scores[name] = torch.rand(self.apts[name].input_dim)
            else:
                salience_scores[name] = torch.rand(self.apts[name].input_dim)
                
        # Update pruning scheduler
        masks = self.pruning_scheduler.update(current_step, salience_scores)
        
        # Apply masks to APT adapters
        for name, mask in masks.items():
            if name in self.apts:
                apt = self.apts[name]
                # Apply masks (simplified)
                if 'head' in name:
                    apt.set_pruning_masks(
                        input_mask=torch.ones(apt.input_dim),
                        output_mask=mask
                    )
                else:
                    apt.set_pruning_masks(
                        input_mask=mask,
                        output_mask=torch.ones(apt.output_dim)
                    )
                    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                if self.args.task == 'alpaca':
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # For language modeling, we'd evaluate perplexity
                    # For simplicity, we'll return 0
                    continue
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
        accuracy = correct / total if total > 0 else 0
        return accuracy
        
    def train(self):
        """Main training loop"""
        logger.info(f"Starting APT training for {self.args.model_type} on {self.args.task}")
        logger.info(f"Target sparsity: {self.args.sparsity}")
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        if train_loader is None:
            logger.error("Failed to load data")
            return
            
        # Training loop
        best_accuracy = 0.0
        training_times = []
        
        for epoch in range(self.args.epochs):
            start_time = time.time()
            
            # Train epoch
            avg_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            if val_loader is not None:
                accuracy = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self._save_model(f"best_model_epoch_{epoch+1}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.args.epochs} - Loss: {avg_loss:.4f}")
                
            training_times.append(time.time() - start_time)
            
        # Save final model
        self._save_model("final_model")
        
        # Print summary
        logger.info(f"Training completed!")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        logger.info(f"Average training time per epoch: {np.mean(training_times):.2f}s")
        
        # Print model sparsity
        self._print_model_sparsity()
        
    def _save_model(self, name: str):
        """Save model and adapters"""
        save_path = os.path.join(self.args.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save APT adapter parameters
        apt_state = {}
        for name, apt in self.apts.items():
            apt_state[name] = {
                'W_A': apt.W_A.data.cpu(),
                'W_B': apt.W_B.data.cpu(),
                'input_mask': apt.input_mask.data.cpu() if apt.input_mask is not None else None,
                'output_mask': apt.output_mask.data.cpu() if apt.output_mask is not None else None,
                'rank': apt.rank
            }
            
        torch.save(apt_state, os.path.join(save_path, 'apt_adapters.pth'))
        
    def _print_model_sparsity(self):
        """Print model sparsity statistics"""
        total_params = 0
        pruned_params = 0
        
        for name, apt in self.apts.items():
            total = apt.get_total_parameters()
            active = apt.get_active_parameters()
            
            total_params += total
            pruned_params += (total - active)
            
        sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Model sparsity: {sparsity:.4f} ({pruned_params}/{total_params} parameters pruned)")
        
def main():
    parser = argparse.ArgumentParser(description='APT: Adaptive Pruning and Tuning')
    
    # Model and task
    parser.add_argument('--model_type', type=str, default='roberta', 
                       choices=['roberta', 't5', 'llama'],
                       help='Model type to use')
    parser.add_argument('--task', type=str, default='sst2',
                       choices=['sst2', 'mnli', 'squad', 'cnndm', 'alpaca'],
                       help='Task to train on')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--sparsity', type=float, default=0.6,
                       help='Target sparsity (0.0 to 1.0)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer and train
    trainer = APTTrainer(args)
    trainer.train()
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
        
if __name__ == "__main__":
    main()