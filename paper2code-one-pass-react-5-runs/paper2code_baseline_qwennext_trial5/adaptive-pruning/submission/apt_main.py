#!/usr/bin/env python3
"""
Main APT implementation
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk
import numpy as np
import random
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import logging

from apt_adapter import APTModel, APTAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APTTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        # Load model and tokenizer
        self.model_name = args.model_name
        self.task = args.task
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Initialize APT model
        self.config = {
            "model_type": self._get_model_type(),
            "device": self.device,
            "initial_rank": args.initial_rank,
            "alpha": args.alpha,
            "sparsity": args.sparsity,
            "tuning_factor": args.tuning_factor
        }
        
        self.ap_model = APTModel(self.model, self.config)
        self.ap_model.to(self.device)
        
        # Prepare dataset
        self.train_loader, self.eval_loader = self._prepare_dataset()
        
        # Optimizer and scheduler
        self.optimizer = AdamW(self.ap_model.parameters(), lr=args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * args.epochs
        )
        
        # Initialize distillation
        self.ap_model.enable_distillation()
        
        # Track metrics
        self.best_eval_score = 0.0
        self.train_losses = []
        self.eval_scores = []
        
    def _get_model_type(self):
        """Determine model type based on model name"""
        if "roberta" in self.model_name.lower():
            return "roberta"
        elif "t5" in self.model_name.lower():
            return "t5"
        elif "llama" in self.model_name.lower():
            return "llama"
        else:
            return "other"
            
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer based on task"""
        if self.task in ["sst2", "mnli", "qqp", "qnli", "mrpc", "cola", "rte", "sts-b"]:
            # Classification tasks
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2 if self.task == "sst2" else 3 if self.task == "mnli" else 2,
                cache_dir="cache"
            )
        elif self.task == "cnn_dailymail":
            # Summarization task
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir="cache")
        elif self.task == "alpaca":
            # Instruction tuning task
            model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir="cache")
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir="cache")
        
        # Set padding token for models that don't have it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    def _prepare_dataset(self):
        """Prepare dataset based on task"""
        if self.task == "sst2":
            dataset = load_from_disk("data/sst2")
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["sentence"],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
        elif self.task == "cnn_dailymail":
            dataset = load_from_disk("data/cnn_dailymail")
            def tokenize_function(examples):
                inputs = [f"summarize: {doc}" for doc in examples["article"]]
                targets = examples["highlights"]
                
                model_inputs = self.tokenizer(
                    inputs,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets,
                        max_length=128,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
                
        elif self.task == "alpaca":
            # Use dummy dataset for reproduction
            if self.args.use_dummy_llama:
                # Load dummy dataset
                dummy_data = torch.load("data/alpaca/dummy_dataset.pt")
                dataset = {
                    "train": {
                        "instruction": dummy_data["instruction"],
                        "input": dummy_data["input"],
                        "output": dummy_data["output"]
                    }
                }
                
                def tokenize_function(examples):
                    inputs = [
                        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
                    ]
                    
                    model_inputs = self.tokenizer(
                        inputs,
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    # Create labels (same as input_ids but with -100 for non-response parts)
                    labels = model_inputs["input_ids"].clone()
                    # We'll set the loss to -100 for the instruction and input parts
                    # This is a simplified version
                    model_inputs["labels"] = labels
                    return model_inputs
                    
            else:
                # In real reproduction, we would load the actual Alpaca dataset
                raise NotImplementedError("Real Alpaca dataset requires access request")
                
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
        # Apply tokenization
        tokenized_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Create data loaders
        train_loader = DataLoader(
            tokenized_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        # For evaluation, use a small subset
        eval_dataset = tokenized_dataset.select(range(min(100, len(tokenized_dataset))))
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )
        
        return train_loader, eval_loader
        
    def _calculate_salience_scores(self, batch):
        """Calculate salience scores for adapters (simplified version)"""
        # In a real implementation, we would capture gradients and activations
        # Here we use a simplified approach based on parameter magnitudes
        
        salience_scores = {}
        
        for adapter_name, adapter in self.ap_model.adapters.items():
            # Simplified salience: sum of absolute values of W_A and W_B weights
            # This is not the same as the paper's method but serves as a placeholder
            w_a_salience = torch.sum(torch.abs(adapter.W_A)).item()
            w_b_salience = torch.sum(torch.abs(adapter.W_B)).item()
            salience_scores[adapter_name] = w_a_salience + w_b_salience
            
        return salience_scores
        
    def _apply_adaptive_pruning(self, epoch):
        """Apply adaptive pruning based on epoch"""
        if epoch < self.args.prune_start_epoch:
            return
            
        # Calculate salience scores
        salience_scores = self._calculate_salience_scores(None)
        
        # Apply pruning
        self.ap_model.adaptive_pruning(salience_scores, self.args.sparsity)
        
        logger.info(f"Epoch {epoch}: Applied adaptive pruning with sparsity {self.args.sparsity}")
        
    def _apply_adaptive_tuning(self, epoch):
        """Apply adaptive tuning based on epoch"""
        if epoch < self.args.tune_start_epoch:
            return
            
        # Calculate salience scores
        salience_scores = self._calculate_salience_scores(None)
        
        # Apply tuning
        self.ap_model.adaptive_tuning(salience_scores, self.args.tuning_factor)
        
        logger.info(f"Epoch {epoch}: Applied adaptive tuning with factor {self.args.tuning_factor}")
        
    def _compute_loss(self, batch, epoch):
        """Compute loss with distillation"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        student_outputs = self.ap_model(**batch)
        
        # Compute supervised loss
        if self.task in ["sst2", "mnli", "qqp", "qnli", "mrpc", "cola", "rte", "sts-b"]:
            # Classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(student_outputs.logits.view(-1, student_outputs.logits.size(-1)), batch["labels"].view(-1))
        elif self.task == "cnn_dailymail":
            # Summarization
            loss = student_outputs.loss
        elif self.task == "alpaca":
            # Instruction tuning
            loss = student_outputs.loss
            
        # Compute distillation loss
        distill_loss = self.ap_model.distillation_loss(student_outputs, student_outputs)
        
        # Combine losses
        total_loss = loss + self.args.distill_weight * distill_loss
        
        return total_loss, loss, distill_loss
        
    def train(self):
        """Main training loop"""
        logger.info("Starting APT training...")
        
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.epochs}")
            
            # Apply adaptive pruning and tuning
            self._apply_adaptive_pruning(epoch)
            self._apply_adaptive_tuning(epoch)
            
            # Training
            self.ap_model.train()
            total_loss = 0.0
            total_supervised_loss = 0.0
            total_distill_loss = 0.0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                self.optimizer.zero_grad()
                
                loss, supervised_loss, distill_loss = self._compute_loss(batch, epoch)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                total_supervised_loss += supervised_loss.item()
                total_distill_loss += distill_loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            avg_supervised_loss = total_supervised_loss / len(self.train_loader)
            avg_distill_loss = total_distill_loss / len(self.train_loader)
            
            self.train_losses.append(avg_loss)
            logger.info(f"  Training Loss: {avg_loss:.4f} (Supervised: {avg_supervised_loss:.4f}, Distill: {avg_distill_loss:.4f})")
            
            # Evaluation
            eval_score = self.evaluate()
            self.eval_scores.append(eval_score)
            
            # Save best model
            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self._save_model(f"{self.args.output_dir}/best_model")
                
            logger.info(f"  Evaluation Score: {eval_score:.4f} (Best: {self.best_eval_score:.4f})")
            
        # Save final model
        self._save_model(f"{self.args.output_dir}/final_model")
        
        # Save metrics
        metrics = {
            "train_losses": self.train_losses,
            "eval_scores": self.eval_scores,
            "best_eval_score": self.best_eval_score,
            "final_sparsity": self.args.sparsity,
            "final_tuning_factor": self.args.tuning_factor
        }
        
        with open(f"{self.args.output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Training completed. Best evaluation score: {self.best_eval_score:.4f}")
        
    def evaluate(self):
        """Evaluate model on validation set"""
        self.ap_model.eval()
        total_score = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.ap_model(**batch)
                
                if self.task in ["sst2", "mnli", "qqp", "qnli", "mrpc", "cola", "rte", "sts-b"]:
                    # Classification accuracy
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct = (predictions == batch["labels"]).float().sum()
                    total_score += correct.item()
                    total_samples += len(batch["labels"])
                    
                elif self.task == "cnn_dailymail":
                    # Use perplexity as proxy for summarization quality
                    loss = outputs.loss
                    total_score += -loss.item()  # Higher is better
                    total_samples += 1
                    
                elif self.task == "alpaca":
                    # Use perplexity as proxy for instruction following
                    loss = outputs.loss
                    total_score += -loss.item()  # Higher is better
                    total_samples += 1
                    
        if total_samples > 0:
            return total_score / total_samples
        else:
            return 0.0
            
    def _save_model(self, path):
        """Save model and adapter weights"""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.ap_model.state_dict(),
            "config": self.config,
            "args": vars(self.args)
        }, f"{path}/model.pt")
        
        # Save adapter information
        adapter_info = {}
        for name, adapter in self.ap_model.adapters.items():
            adapter_info[name] = {
                "input_mask": adapter.input_mask.cpu().numpy().tolist(),
                "output_mask": adapter.output_mask.cpu().numpy().tolist(),
                "current_rank": adapter.current_rank,
                "original_rank": adapter.rank,
                "pruned_input": adapter.pruned_input,
                "pruned_output": adapter.pruned_output
            }
            
        with open(f"{path}/adapters.json", "w") as f:
            json.dump(adapter_info, f, indent=2)
            
        logger.info(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="APT: Adaptive Pruning and Tuning")
    
    # Model and task
    parser.add_argument("--model_name", type=str, default="roberta-base", 
                       help="Model name (e.g., roberta-base, t5-base, meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--task", type=str, default="sst2", 
                       help="Task name (sst2, cnn_dailymail, alpaca)")
    parser.add_argument("--use_dummy_llama", action="store_true", 
                       help="Use dummy dataset for LLaMA (for reproduction)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # APT parameters
    parser.add_argument("--initial_rank", type=int, default=8, help="Initial LoRA rank")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha parameter")
    parser.add_argument("--sparsity", type=float, default=0.6, help="Target sparsity (0.0-1.0)")
    parser.add_argument("--tuning_factor", type=float, default=1.5, help="Tuning factor (increase rank by this factor)")
    parser.add_argument("--prune_start_epoch", type=int, default=1, help="Epoch to start pruning")
    parser.add_argument("--tune_start_epoch", type=int, default=2, help="Epoch to start tuning")
    parser.add_argument("--distill_weight", type=float, default=0.1, help="Weight for distillation loss")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/apt", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and train
    trainer = APTTrainer(args)
    trainer.train()
    
    logger.info(f"APT training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()