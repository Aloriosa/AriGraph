import torch
import math
from transformers import AutoModelForSequenceClassification, AutoConfig

def prune_attention_heads(model, prune_percent):
    """
    Prune a given percentage of attention heads from the model.
    The heads are selected based on the magnitude of the attention
    weight matrix of each transformer block.
    """
    if not hasattr(model, "config"):
        raise ValueError("Model must have a config attribute")

    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    heads_to_prune = math.ceil(num_heads * prune_percent)

    # Collect head scores across all encoder layers
    head_scores = []
    for i, layer in enumerate(model.base_model.encoder.layer):
        # attention.self.query.weight shape: (hidden, hidden)
        query_weight = layer.attention.self.query.weight.data
        # Reshape to (num_heads, head_dim, hidden)
        # Compute L2 norm per head
        query_weight = query_weight.view(num_heads, head_dim, -1)
        head_norm = query_weight.norm(p=2, dim=(1,2))
        head_scores.append(head_norm)

    # Average scores across layers
    mean_scores = torch.stack(head_scores, dim=0).mean(dim=0)

    # Identify heads to prune (lowest scores)
    _, prune_indices = torch.topk(mean_scores, heads_to_prune, largest=False)

    # Use HuggingFace pruning utility
    for layer in model.base_model.encoder.layer:
        # The pruning function expects a list of head indices to prune
        layer.attention.prune_heads(set(prune_indices.tolist()))

    print(f"Pruned {heads_to_prune} heads from each encoder layer.")

def adjust_adapter_rank(adapter, new_rank):
    """
    Dynamically increase the rank of a LoRA/APT adapter by adding
    new randomly initialized rows/columns.  The existing weights
    are preserved.
    """
    old_rank = adapter.rank
    if new_rank <= old_rank:
        return  # nothing to do

    # Expand down projection (in APT: W_B)
    weight_in = adapter.W_B.weight.data
    weight_out = adapter.W_A.weight.data

    # New rows for W_B
    new_rows = torch.randn_like(weight_in[:, old_rank:])
    # New cols for W_A
    new_cols = torch.randn_like(weight_out[old_rank:, :])

    # Concatenate
    adapter.W_B.weight.data = torch.cat([weight_in, new_rows], dim=1)
    adapter.W_A.weight.data = torch.cat([weight_out, new_cols], dim=0)

    adapter.rank = new_rank
    print(f"Increased adapter rank from {old_rank} to {new_rank}.")