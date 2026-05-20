import torch
from torch.utils.data import DataLoader
from src.arch.transformer_score_model import TransformerScoreModel
from src.data.tokenizer import Tokenizer
from src.data.simulator_wrapper import SimulatorWrapper
import numpy as np

class DiffusionTrainer:
    def __init__(
        self,
        model: TransformerScoreModel,
        tokenizer: Tokenizer,
        simulator: SimulatorWrapper,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        batch_size: int = 32,
        max_seq_len: int = 512,
        timesteps: int = 1000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.simulator = simulator
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.timesteps = timesteps
        self.model.to(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Generate new simulated data and masks for this batch
            sim_data = self.simulator.generate_batch(self.batch_size, n_params=10)
            tokens_list = self.tokenizer.encode(sim_data)
            
            # Convert to tensor and pad
            tokens = self._pad_tokens(tokens_list, self.max_seq_len)
            tokens = tokens.to(self.device)

            # Sample random timesteps
            t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).float() / self.timesteps

            # Generate random mask for this iteration (resampled at every iteration)
            mask = self._generate_random_mask(tokens.shape, device=self.device)

            # Compute loss
            loss = self.compute_loss(tokens, mask, t)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def compute_loss(self, x0: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time-dependent denoising score matching loss.
        Uses the formulation from Song et al. (2021b): 
        L(t) = E[ || s_theta(x_t, t) - score(x_t | x_0) ||^2 ]
        where x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        """
        batch_size, seq_len = x0.shape

        # Sample noise
        noise = torch.randn_like(x0, device=x0.device)

        # Compute alpha_t and beta_t (simple linear schedule)
        alpha_t = torch.cos(t * np.pi / 2) ** 2  # cosine schedule for stability
        alpha_t = alpha_t.view(-1, 1).to(x0.device)
        
        # Sample x_t: noisy version of x0
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

        # Compute true score: score(x_t | x_0) = (x_0 - x_t) / (1 - alpha_t)
        true_score = (x0 - x_t) / (1 - alpha_t)

        # Model predicts score function
        predicted_score = self.model(x_t, mask, t)

        # Masked loss: only compute loss on masked positions
        loss = torch.mean((predicted_score - true_score) ** 2 * mask.float())
        
        return loss

    def _pad_tokens(self, tokens_list: list[dict], max_len: int) -> torch.Tensor:
        """Pad list of token dicts to fixed length."""
        # Assuming each token dict has a 'ids' field
        ids_list = [torch.tensor(t['ids'], dtype=torch.long) for t in tokens_list]
        padded = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=0)
        if padded.size(1) > max_len:
            padded = padded[:, :max_len]
        else:
            padded = torch.nn.functional.pad(padded, (0, max_len - padded.size(1)), value=0)
        return padded

    def _generate_random_mask(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate random binary mask with 50% probability per token."""
        batch_size, seq_len = shape
        mask = torch.rand(batch_size, seq_len, device=device) > 0.5
        return mask.float()