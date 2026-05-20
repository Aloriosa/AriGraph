import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load a causal language model and its tokenizer.
    The tokenizer is extended with a special [MEM] token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure the tokenizer has a padding token for GPT2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add the special memory token
    special_tokens_dict = {"additional_special_tokens": ["[MEM]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return tokenizer, model