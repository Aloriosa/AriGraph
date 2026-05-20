"""
Implementation of Classifier-Free Guidance for language models
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Optional

class CFGSampler:
    """Implements Classifier-Free Guidance for language models
    
    Based on the paper "Stay on topic with Classifier-Free Guidance"
    """
    
    def __init__(self, model, tokenizer, gamma=1.5):
        """
        Initialize the CFG sampler
        
        Args:
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            gamma: Guidance strength parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = gamma
        
    def sample(self, prompt: str, max_length: int = 1024, temperature: float = 1.0) -> str:
        """
        Generate text using CFG with given prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate conditional output
        with torch.no_grad():
            outputs_conditional = self.model(input_ids, attention_mask, output_hidden_states=True)
        logits_conditional = outputs_conditional.logits[:, -1, :]
        
        # Generate unconditional output
        # For unconditional, we use an empty prompt
        empty_prompt = ""
        inputs_unconditional = self.tokenizer(empty_prompt, return_tensors="pt")
        input_ids_unconditional = inputs_unconditional["input_ids"]
        attention_mask_unconditional = inputs_unconditional["attention_mask"]
        
        with torch.no_grad():
            outputs_unconditional = self.model(input_ids_unconditional, attention_unconditional, output_hidden_states=True)
        logits_unconditional = outputs_unconditional.logits[:, -1, :]
        
        # Apply CFG formula: logits = logits_conditional + gamma * (logits_conditional - logits_unconditional)
        # This is the core of Classifier-Free Guidance
        logits = logits_conditional + self.gamma * (logits_conditional - logits_unconditional)
        
        # Sample from the modified logits
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # Generate full sequence
        generated_ids = input_ids
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        for _ in range(max_length - generated_ids.shape[1]):
            with torch.no_grad():
                outputs = self.model(generated_ids, attention_mask=torch.ones_like(generated_ids), output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            # Stop if we generate EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text