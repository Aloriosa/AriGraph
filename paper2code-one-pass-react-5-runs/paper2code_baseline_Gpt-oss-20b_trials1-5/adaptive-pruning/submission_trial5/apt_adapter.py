# apt_adapter.py
"""
A minimal implementation of the APT (Adaptive Pruning and Tuning) idea
for BERT-based sequence‑classification models.

The implementation focuses on two key ideas:
    1. LoRA adapters with *dynamic rank* – the rank of each adapter
       can be increased during training.
    2. Structured pruning of attention heads – heads with low salience
       are removed early in training.

This is **not** a drop‑in replacement for the full paper implementation
but a lightweight reproduction that can be executed on a single A10 GPU
within a few minutes.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForSequenceClassification
from typing import List, Dict


class LoRAAdapter(nn.Module):
    """
    A simple LoRA adapter that wraps an existing nn.Linear layer.
    """
    def __init__(self, orig_layer: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.orig = orig_layer
        self.in_dim = orig_layer.in_features
        self.out_dim = orig_layer.out_features
        self.r = r
        self.alpha = alpha
        # Scaling factor
        self.scaling = alpha / r
        # LoRA parameters
        self.A = nn.Parameter(torch.randn(r, self.in_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(self.out_dim, r) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        out = self.orig(x)
        # LoRA low‑rank update
        lora = self.scaling * (self.B @ (self.A @ x.t()))  # shape (out_dim, batch)
        return out + lora.t()


class APTBertForSequenceClassification(BertForSequenceClassification):
    """
    BertForSequenceClassification with:
        • LoRA adapters inserted into query and value projections.
        • Dynamic rank adjustment.
        • Head pruning.
    """
    def __init__(self,
                 config: BertConfig,
                 init_rank: int = 8,
                 max_rank: int = 32,
                 rank_increase_epochs: List[int] = None,
                 target_head_sparsity: float = 0.4,
                 device: torch.device = None):
        super().__init__(config)
        self.device = device or torch.device('cpu')
        self.init_rank = init_rank
        self.max_rank = max_rank
        self.rank_increase_epochs = rank_increase_epochs or [10, 20]
        self.current_rank = init_rank
        self.target_head_sparsity = target_head_sparsity

        # Replace query & value layers with LoRA adapters
        for layer in self.bert.encoder.layer:
            # Query projection
            ori_q = layer.attention.self.query
            layer.attention.self.query = LoRAAdapter(ori_q, r=self.current_rank, alpha=1.0).to(self.device)
            # Value projection
            ori_v = layer.attention.self.value
            layer.attention.self.value = LoRAAdapter(ori_v, r=self.current_rank, alpha=1.0).to(self.device)

        # Keep track of pruned heads
        self.pruned_heads: Dict[int, List[int]] = {i: [] for i in range(config.num_hidden_layers)}

        # Helper to compute head indices to prune
        self.heads_to_prune: List[int] = []

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        # Forward through BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=False)
        # CLS token representation
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits)

    # ------------------------------------------------------------------
    # Methods for dynamic rank adjustment
    # ------------------------------------------------------------------
    def increase_rank(self):
        """
        Increase the rank of all LoRA adapters by one, up to max_rank.
        """
        if self.current_rank >= self.max_rank:
            return
        self.current_rank += 1
        for layer in self.bert.encoder.layer:
            # Query
            lo = layer.attention.self.query
            lo.A.data = torch.cat([lo.A.data, torch.randn(1, lo.in_dim) * 0.02], dim=0)
            lo.B.data = torch.cat([lo.B.data, torch.randn(lo.out_dim, 1) * 0.02], dim=1)
            lo.r = lo.A.size(0)
            # Value
            lo = layer.attention.self.value
            lo.A.data = torch.cat([lo.A.data, torch.randn(1, lo.in_dim) * 0.02], dim=0)
            lo.B.data = torch.cat([lo.B.data, torch.randn(lo.out_dim, 1) * 0.02], dim=1)
            lo.r = lo.A.size(0)

    # ------------------------------------------------------------------
    # Methods for structured head pruning
    # ------------------------------------------------------------------
    def compute_head_salience(self, data_loader, device, num_samples=200):
        """
        Compute a very simple salience score for each head:
            mean absolute gradient of the head output w.r.t. loss.
        """
        self.eval()
        salience = {i: torch.zeros(self.config.num_attention_heads, device=device)
                    for i in range(self.config.num_hidden_layers)}
        loss_fct = nn.CrossEntropyLoss()
        with torch.no_grad():
            for _ in range(num_samples // data_loader.batch_size):
                batch = next(iter(data_loader))
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # Enable gradient on attention outputs
                for layer in self.bert.encoder.layer:
                    layer.attention.self.register_forward_hook(self._store_attn_output)
                outputs = self.forward(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       labels=labels)
                loss, _ = outputs
                loss.backward()

                # Aggregate gradients per head
                for layer_idx, grads in self.attn_grads.items():
                    # grads shape: (batch, seq, heads, head_dim)
                    grad_abs = grads.abs().mean(dim=(0, 1, 3))  # mean over batch, seq, head_dim
                    salience[layer_idx] += grad_abs
                # Clear stored grads
                self.attn_grads.clear()

                if _ == num_samples // data_loader.batch_size - 1:
                    break
        # Average over samples
        for layer_idx in salience:
            salience[layer_idx] /= (num_samples // data_loader.batch_size)
        return salience

    def _store_attn_output(self, module, input, output):
        """
        Hook to store attention outputs for salience computation.
        """
        # output[1] is the attention output (batch, seq, heads, head_dim)
        grads = output[1].detach().requires_grad_(True)
        self.attn_grads.setdefault(module, grads)

    def prune_heads(self, salience: Dict[int, torch.Tensor]):
        """
        Prune heads with lowest salience until target_head_sparsity.
        """
        for layer_idx, head_salience in salience.items():
            num_heads = head_salience.size(0)
            num_to_keep = int(num_heads * (1.0 - self.target_head_sparsity))
            if num_to_keep <= 0:
                continue
            # Keep heads with highest salience
            _, idx = torch.topk(head_salience, num_to_keep)
            prune_idx = [i for i in range(num_heads) if i not in idx.tolist()]
            self.pruned_heads[layer_idx] = prune_idx
            # Apply pruning to the layer
            layer = self.bert.encoder.layer[layer_idx]
            # Zero out query and value weights for pruned heads
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            # Query
            q_weights = layer.attention.self.query.orig.weight.data
            q_bias = layer.attention.self.query.orig.bias.data
            for h in prune_idx:
                start = h * head_dim
                end = start + head_dim
                q_weights[:, start:end] = 0
                q_bias[start:end] = 0
            layer.attention.self.query.orig.weight.data = q_weights
            layer.attention.self.query.orig.bias.data = q_bias
            # Value
            v_weights = layer.attention.self.value.orig.weight.data
            v_bias = layer.attention.self.value.orig.bias.data
            for h in prune_idx:
                start = h * head_dim
                end = start + head_dim
                v_weights[:, start:end] = 0
                v_bias[start:end] = 0
            layer.attention.self.value.orig.weight.data = v_weights
            layer.attention.self.value.orig.bias.data = v_bias