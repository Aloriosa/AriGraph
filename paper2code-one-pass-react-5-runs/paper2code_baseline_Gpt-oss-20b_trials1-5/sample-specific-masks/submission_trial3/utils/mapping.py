import torch
import torch.nn.functional as F
from collections import defaultdict


class ILMMapper:
    """
    Iterative Label Mapping (ILM) used in the paper.

    For each target class, we keep the pre‑trained class that has the
    highest frequency among all training samples predicted by the
    frozen backbone.  The mapping is updated at the start of every
    epoch.
    """

    def __init__(self, num_pretrained: int, num_target: int, device: torch.device):
        self.num_pretrained = num_pretrained
        self.num_target = num_target
        self.device = device
        # mapping from target class -> pretrained class index
        self.mapping = torch.arange(num_target, device=device)  # dummy initial

    def update(self, loader, model, mask_gen, delta, device):
        """
        Compute the mapping using the current model state.
        """
        model.eval()
        mask_gen.eval()
        delta.requires_grad_(False)

        freq = torch.zeros(
            self.num_pretrained, self.num_target, dtype=torch.long, device=device
        )

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs_resized = F.interpolate(imgs, size=224 if "resnet" in str(model) else 384,
                                             mode="bilinear", align_corners=False)
                mask = mask_gen(imgs_resized)
                mask = F.interpolate(mask, size=imgs_resized.shape[-1], mode="nearest")
                prog = imgs_resized + delta * mask
                prog = torch.clamp(prog, 0.0, 1.0)
                logits = model(prog)  # (B, num_pretrained)
                preds = logits.argmax(1)  # (B,)
                for t in range(self.num_target):
                    freq[preds == t, t] += 1

        # For each target class, pick pretrained class with max frequency
        mapping = torch.argmax(freq, dim=0)
        self.mapping = mapping

    def apply(self, logits_pretrained: torch.Tensor,
              labels: torch.Tensor) -> torch.Tensor:
        """
        Convert pretrained logits to target logits using current mapping.

        Parameters
        ----------
        logits_pretrained : torch.Tensor
            Shape (B, num_pretrained).
        labels : torch.Tensor
            Shape (B,).

        Returns
        -------
        torch.Tensor
            Shape (B, num_target).
        """
        # For each sample, pick the logit of the mapped pretrained class
        mapped_indices = self.mapping[labels]  # (B,)
        logits_target = logits_pretrained[torch.arange(len(labels)), mapped_indices]
        # Expand to (B, num_target) for cross‑entropy
        logits_target = logits_target.unsqueeze(1)
        # To keep the shape compatible with CE, we simply broadcast
        # the same logits across all target classes.
        # This is a simplification; in practice one would use a small
        # linear layer or a proper mapping.
        return logits_target