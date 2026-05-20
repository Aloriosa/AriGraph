"""
Utilities for loading and running models
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Optional

class ModelUtils:
    """Utilities for loading and running models"""
    
    @staticmethod
    def load_model(model_name: str, model_path: str):
        """Load a model from path"""
        if model_name == 'gpt2-medium':
            model = GPT2LMHeadModel.from_pretrained(model_path)
        elif model_name == 'gpt2-large':
            model = GPT2LMHeadModel.from_pretrained(model_path)
        elif model_name == 'llama-7b':
            model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Model {model_name} not supported")
        model.eval()
        return model
    
    @staticmethod
    def load_tokenizer(model_name: str):
        """Load tokenizer"""
        if model_name == 'gpt2-medium':
            return GPT2Tokenizer.from_pretrained('gpt2-medium')
        elif model_name == 'gpt2-large':
            return GPT2Tokenizer.from_pretrained('gpt2-large')
        elif model_name == 'llama-7b':
            return GPT2Tokenizer.from_pretrained('gpt2-medium')
        else:
            raise ValueError(f"Model {model_name} not supported"