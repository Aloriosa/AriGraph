import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_generation_model(model_name: str = "gpt2", device: str = None):
    """
    Load a causal language model and its tokenizer.
    If device is None, automatically use GPU if available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, device