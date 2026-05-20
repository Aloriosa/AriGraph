import torch
import torch.nn as nn
import math

class APTLinear(nn.Module):
    """
    A linear layer with:
      - Binary input/output masks for structured pruning.
      - Low-rank LoRA adaptation with dynamic rank.
    """
    def __init__(self, in_features, out_features, bias=True,
                 rank=8, init_scale=0.01, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Original weight and bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Masks for pruning
        self.mask_in = nn.Parameter(torch.ones(in_features, dtype=torch.bool, device=device))
        self.mask_out = nn.Parameter(torch.ones(out_features, dtype=torch.bool, device=device))

        # LoRA parameters
        self.rank = rank
        self.A = nn.Parameter(torch.randn((rank, in_features), device=device) * init_scale)
        self.B = nn.Parameter(torch.randn((out_features, rank), device=device) * init_scale)

        # Scaling factor (as in LoRA)
        self.scaling = 1.0

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)
        """
        # Apply masks
        mask_in = self.mask_in.float().unsqueeze(0).unsqueeze(0)   # (1,1,in_features)
        mask_out = self.mask_out.float().unsqueeze(1).unsqueeze(2)  # (out_features,1,1)

        # Compute base linear with masked weight
        masked_weight = self.weight * mask_out  # (out_features, in_features)
        base = torch.matmul(x, masked_weight.t())

        # LoRA contribution
        # x: (batch, seq_len, in_features)
        # A: (rank, in_features), B: (out_features, rank)
        lora = torch.matmul(torch.matmul(x, self.A.t()), self.B.t()) * self.scaling
        lora = lora * mask_out  # ensure pruned outputs are zero

        out = base + lora
        if self.bias is not None:
            out = out + self.bias
        return out

    def prune_neurons(self, prune_ratio):
        """
        Prune a fraction of output neurons based on salience (abs(weight * grad)).
        prune_ratio: fraction of neurons to prune in this call.
        """
        if not self.weight.grad is None:
            # Salience per output neuron
            salience = torch.abs(self.weight * self.weight.grad)
            salience_per_neuron = salience.sum(dim=1)  # (out_features,)
            # Consider only currently unpruned neurons
            active_mask = self.mask_out
            salience_per_neuron = salience_per_neuron * active_mask.float()
            # Number of neurons to prune
            n_to_prune = int(prune_ratio * self.out_features * active_mask.float().sum().item())
            if n_to_prune <= 0:
                return
            # Get indices of neurons with lowest salience
            _, idx = torch.topk(salience_per_neuron, n_to_prune, largest=False)
            # Set their masks to zero
            self.mask_out[idx] = False
            # Zero out corresponding rows in LoRA matrices to avoid gradient accumulation
            self.A.data[:, idx] = 0
            self.B.data[idx, :] = 0
            # Also zero out the weight rows
            self.weight.data[idx, :] = 0

    def increase_rank(self, new_rank):
        """
        Increase LoRA rank by adding new random columns.
        """
        if new_rank <= self.rank:
            return
        add = new_rank - self.rank
        device = self.weight.device
        A_new = nn.Parameter(torch.randn((add, self.in_features), device=device) * 0.01)
        B_new = nn.Parameter(torch.randn((self.out_features, add), device=device) * 0.01)
        # Concatenate
        self.A = nn.Parameter(torch.cat([self.A, A_new], dim=0))
        self.B = nn.Parameter(torch.cat([self.B, B_new], dim=1))
        self.rank = new_rank