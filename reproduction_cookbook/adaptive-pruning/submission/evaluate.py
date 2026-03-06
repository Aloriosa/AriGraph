import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
import numpy as np
import json
import os
from tqdm import tqdm

class APTEvaluator:
    """
    Evaluator class for APT models.
    """
    
    def __init__(self, model_name: str, task: str, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.task = task
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
    def evaluate(self, dataloader, output_file: str = None) -> Dict[str, float]:
        """Evaluate model performance"""
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                    
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
                total_loss += loss.item()
                
                # For classification tasks
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    predictions_batch = torch.argmax(logits, dim=-1)
                    labels = batch['labels']
                    
                    correct += (predictions_batch == labels).sum().item()
                    total += labels.size(0)
                    
                    predictions.extend(predictions_batch.cpu().tolist())
                    references.extend(labels.cpu().tolist())
                    
                # For generation tasks
                elif hasattr(outputs, 'sequences'):
                    # For generation tasks
                    generated_sequences = outputs.sequences
                    # In practice, we'd compare with references
                    # For now, we'll just store the generated sequences
                    predictions.extend(generated_sequences.cpu().tolist())
                    references.extend(batch['labels'].cpu().tolist())
                    
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate additional metrics based on task
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        # For classification tasks, calculate F1 score
        if self.task in ['sst2', 'mrpc', 'cola', 'rte', 'qnli', 'mnli', 'qqp']:
            from sklearn.metrics import f1_score
            f1 = f1_score(references, predictions, average='weighted')
            metrics['f1'] = f1
            
        # For generation tasks, calculate BLEU score
        elif self.task in ['cnndm', 'xsum', 'alpaca']:
            # BLEU score calculation would go here
            # For now, we'll just use accuracy as a proxy
            pass
            
        # Save results if output file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        return metrics
        
    def evaluate_on_tasks(self, task_dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Dict[str, float]]:
        """Evaluate on multiple tasks"""
        results = {}
        for task_name, dataloader in task_dataloaders.items():
            print(f"Evaluating on {task_name}...")
            metrics = self.evaluate(dataloader)
            results[task_name] = metrics
            
        return results
        
    def save_results(self, results: Dict[str, Dict[str, float]], output_file: str):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    def get_model_size(self) -> int:
        """Get model size in bytes"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4  # Assuming float32 (4 bytes per parameter)
        
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            return memory_mb
        else:
            return 0.0
            
    def get_inference_time(self, dataloader, num_batches: int = 10) -> float:
        """Get average inference time per batch"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        total_time = 0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                    
                start_time.record()
                outputs = self.model(**batch)
                end_time.record()
                
                torch.cuda.synchronize()
                total_time += start_time.elapsed_time(end_time)
                count += 1
                
        return total_time / count if count > 0 else 0.0