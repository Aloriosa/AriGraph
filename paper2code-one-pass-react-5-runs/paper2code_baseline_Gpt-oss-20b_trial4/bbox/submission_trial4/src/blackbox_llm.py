import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class BlackBoxLLM:
    """
    A black‑box LLM wrapper that only exposes a generate method.
    Internally it uses GPT‑2, but from the adapter's point of view
    nothing about gradients or token probabilities is accessible.
    """
    def __init__(self, name: str, device: torch.device, generation_cfg: dict):
        self.device = device
        self.tokenizer = GPT2TokenizerFast.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).to(device)
        self.generation_cfg = generation_cfg

    def generate(self, prompt: str, num_candidates: int):
        """
        Generate `num_candidates` completions for the given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # We use the same prompt for all candidates
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_cfg.get("max_length", 50),
            do_sample=True,
            top_k=self.generation_cfg.get("top_k", 50),
            temperature=self.generation_cfg.get("temperature", 0.7),
            num_return_sequences=num_candidates,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Decode and strip the prompt part
        texts = []
        for out in outputs:
            txt = self.tokenizer.decode(out, skip_special_tokens=True)
            # Remove the original prompt
            txt = txt[len(prompt):].strip()
            texts.append(txt)
        return texts