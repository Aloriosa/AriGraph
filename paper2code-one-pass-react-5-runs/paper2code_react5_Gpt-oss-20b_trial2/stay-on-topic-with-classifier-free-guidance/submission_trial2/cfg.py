"""
Simple implementation of Classifier‑Free Guidance (CFG) for
autoregressive language models (e.g. GPT‑2, Pythia, LLaMA).
Supports:
  * vanilla CFG (γ = 1.0 → no guidance)
  * negative prompting (γ > 0, negative_prompt supplied)
  * top‑k / nucleus filtering
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CFGGenerator:
    """
    Minimal CFG generator.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. "gpt2", "EleutherAI/gpt-neo-2.7B",
        "meta-llama/Llama-2-7b-hf").
    device : str | None
        Device to run on.  If None, the code will automatically choose CUDA if
        available.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        # Ensure the tokenizer has a BOS token for models that require it
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token or self.tokenizer.pad_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """
        Apply top‑k and nucleus (top‑p) filtering.
        """
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            min_val = topk_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift the indices to keep the first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")

        return logits

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        max_new_tokens: int = 50,
        gamma: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> str:
        """
        Generate text from the prompt using CFG sampling.

        Parameters
        ----------
        prompt : str
            Text prompt.
        negative_prompt : str | None
            Text used as the negative conditioning (¬c).  If provided,
            the model will use the logits of `negative_prompt` as the base.
        max_new_tokens : int
            Number of tokens to generate.
        gamma : float
            Guidance weight (γ).  γ=1.0 → vanilla sampling.
        temperature : float
            Sampling temperature.
        top_k : int
            Top‑k filtering.  Set to 0 to disable.
        top_p : float
            Nucleus filtering.  Set to 0.0 to disable.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        str
            Generated text (prompt + continuation).
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Tokenise the prompts
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        if negative_prompt is not None:
            neg_ids = self.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            neg_ids = None

        # Prepare the sequences that will be fed to the model
        # `cond_ids` contains the prompt + generated tokens
        cond_ids = prompt_ids.clone()
        # `uncond_ids` contains the negative prompt or empty if none
        uncond_ids = neg_ids.clone() if neg_ids is not None else torch.empty((1, 0), dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Conditional logits (prompt + generated so far)
                cond_logits = self.model(cond_ids).logits[:, -1, :]

                # Unconditional/negative logits
                if uncond_ids.size(1) == 0:
                    # If no negative prompt, use the model's prior (first token)
                    uncond_logits = torch.zeros_like(cond_logits)
                else:
                    uncond_logits = self.model(uncond_ids).logits[:, -1, :]

                # CFG re‑weighting
                # new_logits = uncond_logits + γ * (cond_logits - uncond_logits)
                new_logits = uncond_logits + gamma * (cond_logits - uncond_logits)
                new_logits = new_logits / temperature
                new_logits = self._apply_filtering(new_logits, top_k, top_p)

                # Sampling
                probs = F.softmax(new_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update sequences
                cond_ids = torch.cat((cond_ids, next_token), dim=1)
                uncond_ids = torch.cat((uncond_ids, next_token), dim=1)

        return self.tokenizer.decode(cond_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Next‑token prediction (for evaluation)
    # ------------------------------------------------------------------
    def predict_next_token(
        self,
        context: str,
        negative_prompt: str | None = None,
        gamma: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> int:
        """
        Return the most‑likely next token ID for the given context using CFG.
        """
        with torch.no_grad():
            ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)

            cond_logits = self.model(ctx_ids).logits[:, -1, :]

            if negative_prompt is None:
                uncond_logits = torch.zeros_like(cond_logits)
            else:
                neg_ids = self.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(self.device)
                uncond_logits = self.model(neg_ids).logits[:, -1, :]

            new_logits = uncond_logits + gamma * (cond_logits - uncond_logits)
            new_logits = self._apply_filtering(new_logits, top_k, top_p)

            return torch.argmax(new_logits, dim=-1).item()