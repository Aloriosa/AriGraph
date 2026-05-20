import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DPOTrainer:
    def __init__(self, model_name, learning_rate=5e-6, batch_size=16, num_epochs=1, beta=0.1, max_length=512, device="cuda"):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta = beta
        self.max_length = max_length
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Move model to device
        self.model.to(device)
        
        # Reference model (frozen copy of original model)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference_model.to(device)
        self.reference_model.eval()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Loss function for DPO
        self.dpo_loss = DPO_Loss(beta=beta)
    
    def tokenize_pair(self, prompt, chosen, rejected):
        """Tokenize a prompt-chosen-rejected triplet"""
        # Format prompt as specified in paper_card_0005
        prompt_formatted = f"[inst] {prompt} [/inst]"
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt_formatted,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Tokenize chosen and rejected responses
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Combine prompt and response
        chosen_input_ids = torch.cat([prompt_tokens.input_ids, chosen_tokens.input_ids], dim=1)
        chosen_attention_mask = torch.cat([prompt_tokens.attention_mask, chosen_tokens.attention_mask], dim=1)
        
        rejected_input_ids = torch.cat([prompt_tokens.input_ids, rejected_tokens.input_ids], dim=1)
        rejected_attention_mask = torch.cat([prompt_tokens.attention_mask, rejected_tokens.attention_mask], dim=1)
        
        return {
            "chosen_input_ids": chosen_input_ids.squeeze(0),
            "chosen_attention_mask": chosen_attention_mask.squeeze(0),
            "rejected_input_ids": rejected_input_ids.squeeze(0),
            "rejected_attention_mask": rejected_attention_mask.squeeze(0),
            "prompt_length": prompt_tokens.input_ids.shape[1]
        }
    
    def prepare_dataset(self, dataset_path):
        """Load and prepare the pairwise preference dataset"""
        with open(dataset_path, "rb") as f:
            pairs = pickle.load(f)
        
        # Tokenize all pairs
        tokenized_data = []
        for pair in pairs:
            tokenized = self.tokenize_pair(pair["prompt"], pair["chosen"], pair["rejected"])
            tokenized_data.append(tokenized)
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(tokenized_data)
        return dataset
    
    def dpo_loss_function(self, batch):
        """Compute DPO loss for a batch of data"""
        # Extract batch elements
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        prompt_length = batch["prompt_length"].to(self.device)
        
        # Get log probabilities for chosen and rejected responses
        with torch.no_grad():
            # Reference model log probabilities
            ref_chosen_logits = self.reference_model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
            ref_rejected_logits = self.reference_model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
            
            # Calculate log probabilities for chosen responses
            ref_chosen_log_probs = torch.gather(ref_chosen_logits, dim=-1, index=chosen_input_ids.unsqueeze(-1)).squeeze(-1)
            ref_chosen_log_probs = (ref_chosen_log_probs * chosen_attention_mask).sum(dim=-1)
            
            # Calculate log probabilities for rejected responses
            ref_rejected_log_probs = torch.gather(ref_rejected_logits, dim=-1, index=rejected_input_ids.unsqueeze(-1)).squeeze(-1)
            ref_rejected_log_probs = (ref_rejected_log_probs * rejected_attention_mask).sum(dim=-1)
        
        # Model log probabilities
        model_chosen_logits = self.model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
        model_rejected_logits = self.model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
        
        # Calculate log probabilities for chosen responses
        model_chosen_log_probs = torch.gather(model_chosen_logits, dim=-1, index=chosen_input_ids.unsqueeze(-1)).squeeze(-1)
        model_chosen_log_probs = (model_chosen_log_probs * chosen_attention_mask).sum(dim=-1)
        
        # Calculate log probabilities for rejected responses
        model_rejected_log_probs = torch.gather(model_rejected_logits, dim=-1, index=rejected_input_ids.unsqueeze(-1)).squeeze(-1)
        model_rejected_log_probs = (model_rejected_log_probs * rejected_attention_mask).sum(dim=-1)
        
        # Compute DPO loss
        # DPO loss: -log(sigmoid(beta * (log_p_chosen - log_p_rejected - (log_p_ref_chosen - log_p_ref_rejected))))
        log_odds_ratio = (model_chosen_log_probs - model_rejected_log_probs) - (ref_chosen_log_probs - ref_rejected_log_probs)
        loss = -torch.nn.functional.logsigmoid(self.beta * log_odds_ratio).mean()
        
        return loss
    
    def train(self, dataset_path, output_dir):
        """Train the model using DPO"""
        logger.info("Loading dataset...")
        dataset = self.prepare_dataset(dataset_path)
        
        logger.info(f"Training dataset size: {len(dataset)}")
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # Compute loss
                loss = self.dpo_loss_function(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if num_batches % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}, Batch {num_batches}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        return total_loss / num_batches
    
    def collate_fn(self, batch):
        """Custom collate function for DPO dataset"""
        # Get the maximum sequence length in the batch
        max_len = max([
            len(item["chosen_input_ids"]) for item in batch
        ] + [
            len(item["rejected_input_ids"]) for item in batch
        ])
        
        # Pad sequences
        padded_batch = {}
        for key in ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]:
            padded_batch[key] = []
            for item in batch:
                seq = item[key]
                padding_length = max_len - len(seq)
                if padding_length > 0:
                    if "input_ids" in key:
                        padded_seq = torch.cat([seq, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=seq.dtype)])
                    else:  # attention_mask
                        padded_seq = torch.cat([seq, torch.zeros(padding_length, dtype=seq.dtype)])
                else:
                    padded_seq = seq
                padded_batch[key].append(padded_seq)
            
            # Stack tensors
            padded_batch[key] = torch.stack(padded_batch[key])
        
        # Handle prompt_length separately
        padded_batch["prompt_length"] = torch.stack([item["prompt_length"] for item in batch])
        
        return padded_batch

class DPO_Loss(nn.Module):
    def __init__(self, beta=0.1):
        super(DPO_Loss, self).__init__()
        self.beta = beta
    
    def forward(self, model_chosen_log_probs, model_rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs):
        # Compute log odds ratio
        log_odds_ratio = (model_chosen_log_probs - model_rejected_log_probs) - (ref_chosen_log_probs - ref_rejected_log_probs)
        
        # Compute DPO loss
        loss = -torch.nn.functional.logsigmoid(self.beta * log_odds_ratio).mean()
        
        return loss

def main():
    parser = argparse.ArgumentParser(description="Train DPO model for toxicity reduction")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the base model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the pairwise preference dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO loss")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        beta=args.beta,
        max_length=args.max_length,
        device=args.device
    )
    
    # Train model
    trainer.train(args.dataset_path, args.output_dir)

if __name__ == "__main__":
    main()