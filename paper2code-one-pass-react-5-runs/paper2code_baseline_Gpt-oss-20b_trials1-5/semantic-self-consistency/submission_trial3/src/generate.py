import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import random


class ChainOfThoughtGenerator:
    """
    Generates a list of chain‑of‑thought (CoT) completions for a given question.
    Uses a causal language model (default GPT‑2) with temperature sampling.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        if device:
            self.model.to(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def _build_prompt(self, question: str) -> str:
        """
        Build a simple chain‑of‑thought prompt.
        """
        return f"Answer the following question step by step.\nQ: {question}\nA:"

    def generate(
        self,
        question: str,
        n: int = 10,
        seed: int | None = None,
    ) -> List[str]:
        """
        Generate `n` CoT completions for the given question.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        prompt = self._build_prompt(question)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and strip the prompt from each completion
        completions = [
            self.tokenizer.decode(out, skip_special_tokens=True)[len(prompt) :].strip()
            for out in outputs
        ]
        return completions