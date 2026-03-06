import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
import numpy as np
import os
from tqdm import tqdm

class APTTrainer:
    """
    Main trainer class for APT method.
    Integrates adaptive pruning, adaptive tuning, and self-distillation.
    """
    
    def __init__(self, model_name: str, task: str, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.task = task
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Initialize APT components
        self.apr = APTAdapter(self.base_model.config.hidden_size, 
                             self.base_model.config.hidden_size, 
                             r=8, alpha=16.0, pruning_ratio=0.6)
        
        # Initialize salience scorer
        self.salience_scorer = OutlierAwareSalienceScorer(self.base_model, self.device)
        
        # Initialize self-distillation
        self.distillation = SelfKnowledgeDistillation(self.base_model)
        
        # Move model to device
        self.base_model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.base_model.parameters(), lr=2e-5)
        
        # Training parameters
        self.max_seq_length = 128
        self.batch_size = 32
        self.num_epochs = 3
        self.pruning_steps = 1000
        self.distillation_steps = 1000
        self.current_step = 0
        
        # Pruning schedule parameters
        self.target_sparsity = 0.6
        self.pruning_schedule = "cubic"
        
    def train(self, train_dataloader, eval_dataloader):
        """Main training loop for APT"""
        self.base_model.train()
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                    
                # Forward pass
                outputs = self.base_model(**batch)
                loss = outputs.loss
                
                # Compute salience scores every few steps
                if self.current_step % 100 == 0:
                    # Compute salience scores
                    salience_scores = self.salience_scorer.compute_salience_scores(train_dataloader, num_batches=5)
                    
                    # Update pruning masks
                    self._update_pruning_masks(salience_scores)
                    
                    # Update tuning ranks
                    self._update_tuning_ranks(salience_scores)
                    
                # Apply self-distillation
                if self.current_step < self.distillation_steps:
                    # Get teacher outputs (using the same model)
                    with torch.no_grad():
                        teacher_outputs = self.base_model(**batch)
                        
                    # Compute distillation loss
                    distillation_loss = self.distillation.get_distillation_loss(outputs.logits, teacher_outputs.logits)
                    distillation_weight = min(self.current_step / self.distillation_steps, 1.0)
                    loss = loss + distillation_weight * distillation_loss
                    
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update step counter
                self.current_step += 1
                
                # Update distillation weight
                self.distillation.update_distillation_weight(self.current_step, self.distillation_steps)
                
                # Update pruning masks
                if self.current_step % 100 == 0:
                    self._update_pruning_masks(salience_scores)
                    
            # Evaluate after each epoch
            self.evaluate(eval_dataloader)
            
    def _update_pruning_masks(self, salience_scores: Dict[str, torch.Tensor]):
        """Update pruning masks based on salience scores"""
        # Update pruning masks for all layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Get salience scores for this layer
                if name in salience_scores:
                    salience = salience_scores[name]
                    
                    # Update pruning mask
                    # This is a simplified version - in practice, we'd use the full APT algorithm
                    if hasattr(module, 'prune_mask'):
                        # Update mask based on salience
                        threshold = torch.quantile(salience, 1 - self.target_sparsity)
                        module.prune_mask = (salience > threshold).float()
                        
    def _update_tuning_ranks(self, salience_scores: Dict[str, torch.Tensor]):
        """Update tuning ranks based on layer salience"""
        # Find the most salient layers and increase their rank
        salience_list = [(name, score.mean().item()) for name, score in salience_scores.items()]
        salience_list.sort(key=lambda x: x[1], reverse=True)
        
        # Increase rank for top 20% of salient layers
        top_layers = salience_list[:int(len(salience_list) * 0.2)]
        
        for name, score in top_layers:
            # Find the corresponding module
            for module_name, module in self.base_model.named_modules():
                if module_name == name and hasattr(module, 'lora_A'):
                    # Increase rank
                    if hasattr(module, 'current_rank'):
                        module.current_rank = min(module.max_rank, module.current_rank + 1)
                        
    def evaluate(self, eval_dataloader):
        """Evaluate model performance"""
        self.base_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                    
                outputs = self.base_model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # For classification tasks
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    labels = batch['labels']
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = correct / total if total > 0 else 0
        
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
        
    def save_model(self, output_dir: str):
        """Save model and training state"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        torch.save({
            'step': self.current_step,
            'target_sparsity': self.target_sparsity,
            'distillation_weight': self.distillation.distillation_weight,
        }, os.path.join(output_dir, 'training_state.pt'))
        
        print(f"Model saved to {output_dir}")
        
    def load_model(self, model_dir: str):
        """Load model and training state"""
        self.base_model = AutoModel.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load training state
        training_state = torch.load(os.path.join(model_dir, 'training_state.pt'))
        self.current_step = training_state['step']
        self.target_sparsity = training_state['target_sparsity']
        self.distillation.distillation_weight = training_state['distillation_weight']
        
        self.base_model.to(self.device)
        
        print(f"Model loaded from {model_dir}")