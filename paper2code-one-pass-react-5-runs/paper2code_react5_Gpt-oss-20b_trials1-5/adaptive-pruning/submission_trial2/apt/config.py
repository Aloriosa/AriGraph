"""
Configuration constants used in the training and evaluation loops.
"""

import math
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model and training
    model_name: str = "distilbert-base-uncased"
    task: str = "sst2"  # or "mnli"
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # LoRA
    lora_init_rank: int = 4
    lora_max_rank: int = 16  # target rank after training

    # Pruning
    target_sparsity: float = 0.4  # final fraction of heads kept
    pruning_steps: int = 20  # number of pruning updates

    # Self-distillation
    distill_start_epoch: int = 1
    distill_end_epoch: int = 3
    distill_weight: float = 0.5  # weight of distillation loss

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed: int = 42