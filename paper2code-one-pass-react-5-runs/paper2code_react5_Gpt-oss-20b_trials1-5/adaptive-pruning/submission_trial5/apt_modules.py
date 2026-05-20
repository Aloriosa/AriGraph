"""
Utilities for LoRA adapters with input/output masking,
dynamic rank growth, and outlier‑aware salience calculation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRAMaskedLinear(nn.Module):
    """
    A linear layer with:
    * frozen original weights (buffer)
    * LoRA low‑rank update (A, B)
    * input‑ and output‑dimension masks (binary masks stored as float for easy multiplication)
    """

    def __init__(self, orig_linear: nn.Linear, rank: int = 4, scaling: float = 1.0):
        super().__init__()
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features

        # Frozen original weights as buffers
        self.register_buffer("W_orig", orig_linear.weight.clone())
        if orig_linear.bias is not None:
            self.register_buffer("bias_orig", orig_linear.bias.clone())
        else:
            self.register_buffer("bias_orig", torch.zeros(out_features))

        # LoRA parameters (trainable)
        self.rank = rank
        self.scaling = scaling
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

        # Input and output masks (binary, but stored as float for easy multiplication)
        self.register_buffer("mask_in", torch.ones(in_features))
        self.register_buffer("mask_out", torch.ones(out_features))

        # Hook to capture gradient w.r.t. output
        self.grad_output = None
        self.register_backward_hook(self._save_grad)

        # Hook to capture activation for kurtosis calculation
        self.activation = None
        self.register_forward_hook(self._save_act)

    def _save_grad(self, module, grad_input, grad_output):
        # grad_output is a tuple with one element: the gradient of the output
        self.grad_output = grad_output[0].detach()

    def _save_act(self, module, input, output):
        # Store the activation before the linear transformation
        self.activation = input[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input mask
        x_masked = x * self.mask_in

        # Effective frozen weight after output mask
        W_eff = self.W_orig * self.mask_out[:, None] * self.mask_in[None, :]

        # Original linear part (frozen)
        out = F.linear(x_masked, W_eff, self.bias_orig)

        # LoRA update
        lora = (x_masked @ self.A.t()) @ self.B.t()
        lora = lora * self.mask_out[:, None]  # apply output mask
        out = out + self.scaling * lora

        return out

    def increase_rank(self, new_rank: int):
        """
        Increase LoRA rank by concatenating new random rows/columns.
        """
        if new_rank <= self.rank:
            return
        # Current parameters
        A_old = self.A.data
        B_old = self.B.data
        in_feat = A_old.shape[1]
        out_feat = B_old.shape[0]

        # New parameters
        A_new = torch.randn(new_rank - self.rank, in_feat) * 0.01
        B_new = torch.randn(out_feat, new_rank - self.rank) * 0.01

        # Concatenate
        self.A = nn.Parameter(torch.cat([A_old, A_new], dim=0))
        self.B = nn.Parameter(torch.cat([B_old, B_new], dim=1))
        self.rank = new_rank

    def mask_prune(self, keep_mask_out: torch.BoolTensor, keep_mask_in: torch.BoolTensor = None):
        """
        Apply boolean masks to the output and (optionally) input dimensions.
        keep_mask_out: shape (out_features,), True for kept dimensions.
        keep_mask_in: shape (in_features,), True for kept dimensions.
        """
        self.mask_out = keep_mask_out.to(self.mask_out.dtype)
        if keep_mask_in is not None:
            self.mask_in = keep_mask_in.to(self.mask_in.dtype)

    def salience(self) -> torch.Tensor:
        """
        Return salience per output dimension: sum of absolute gradients
        plus sqrt of kurtosis of activations.
        If no backward hook has run yet, return zeros.
        """
        if self.grad_output is None or self.activation is None:
            return torch.zeros(self.mask_out.shape[0], device=self.W_orig.device)

        grad_abs_sum = self.grad_output.abs().sum(dim=0).detach()

        # Kurtosis of activations: (E[(x-μ)^4]) / (E[(x-μ)^2])^2
        act = self.activation
        mean = act.mean(dim=0)
        var = act.var(dim=0, unbiased=False) + 1e-6
        kurt = ((act - mean).pow(4).mean(dim=0) + 1e-6) / (var.pow(2))
        # Outlier-aware term: sqrt(kurtosis)
        sal = grad_abs_sum + torch.sqrt(kurt)
        return sal