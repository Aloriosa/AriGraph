"""
Core Classifier‑Free Guidance (CFG) inference logic.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Tuple


class CFGModel:
    """
    Wraps a causal language model to perform CFG‑guided generation.
    """
    def __init__(
        self,
        model_name: str,
        gamma: float = 1.5,
        device: Optional[str] = None,
        use_fast_tokenizer: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            gamma: Guidance strength (γ). γ=1.0 is vanilla
            device: 'cuda' or 'cpu'. If None, auto‑detect
            use_fast_tokenizer: Whether to use the fast tokenizer
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=use_fast_tokenizer
        )
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    def _adjust_logits(self, cond_logits: torch.Tensor, uncond_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the CFG adjustment:
            logit = logit_uncond + γ * (logit_cond - logit_uncond)
        """
        return uncond_logits + self.gamma * (cond_logits - uncond_logits)

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text with CFG.

        Args:
            prompt: Input prompt string (the conditioning part)
            max_length: Number of tokens to generate (after the prompt)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top‑k sampling
            do_sample: Whether to sample or take argmax
            seed: Random seed for reproducibility

        Returns:
            Generated string (prompt + continuation)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = torch.tensor([prompt_ids], device=self.device)

        # Generated tokens (after the prompt)
        gen_ids = torch.tensor([[]], device=self.device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_length):
                # Conditioned logits: prompt + generated tokens
                cond_input = torch.cat([prompt_ids, gen_ids], dim=1)
                cond_logits = self.model(cond_input).logits[:, -1, :]

                # Unconditioned logits: generated tokens only (no prompt)
                if gen_ids.shape[1] == 0:
                    # If no tokens generated yet, use empty input
                    uncond_logits = self.model(torch.tensor(
                        [ [] ], device=self.device, dtype=torch.long
                    )).logits[:, -1, :]
                else:
                    uncond_logits = self.model(gen_ids).logits[:, -1, :]

                # CFG adjustment
                logits = self._adjust_logits(cond_logits, uncond_logits)

                # Apply temperature
                logits = logits / temperature

                # Filter with top‑k / top‑p
                if top_k > 0:
                    logits = torch.topk(logits, top_k, dim=-1).values
                    logits_mask = torch.full_like(logits, float('-inf'))
                    logits = torch.cat([logits_mask, logits], dim=-1)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')

                # Sample or argmax
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append token
                gen_ids = torch.cat([gen_ids, next_token], dim=1)

        # Decode the generated part
        continuation = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return prompt + continuation