import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Tuple

class FRETrainer:
    """
    Trains the encoder and decoder jointly on sampled reward functions.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 dataset: dict,
                 device: torch.device,
                 K_enc: int = 8,
                 K_dec: int = 4,
                 lr: float = 1e-3,
                 beta: float = 0.01,
                 batch_size: int = 64,
                 rng: np.random.Generator = None):
        self.enc = encoder.to(device)
        self.dec = decoder.to(device)
        self.device = device
        self.K_enc = K_enc
        self.K_dec = K_dec
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.rng = rng or np.random.default_rng(0)

        # Optimizer over both encoder and decoder parameters
        self.optimizer = torch.optim.Adam(list(self.enc.parameters()) +
                                         list(self.dec.parameters()),
                                         lr=self.lr)

        # Dataset buffers
        self.obs = torch.tensor(dataset["obs"]).to(device)          # (N, state_dim)
        self.n = self.obs.shape[0]

    def sample_reward_func(self) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
        """
        Returns a callable reward function and a string label.
        Three families are sampled uniformly:
        1. Goal reward: reward = -1 until goal reached.
        2. Linear reward: reward = w · s
        3. MLP reward: reward = tanh(MLP(s))
        """
        choice = self.rng.choice(3)
        if choice == 0:
            # Goal reward
            goal_state = self.rng.choice(self.obs.numpy(), size=1)[0]
            def func(s):
                return -1.0 * (np.linalg.norm(s - goal_state, axis=-1) > 0.2).astype(float)
            return func, "goal"
        elif choice == 1:
            # Linear reward
            w = self.rng.uniform(-1, 1, size=self.obs.shape[1]).astype(np.float32)
            def func(s):
                return (s * w).sum(axis=-1)
            return func, "linear"
        else:
            # MLP reward
            hidden = self.rng.normal(0, 1, size=(self.obs.shape[1], 32)).astype(np.float32)
            out = self.rng.normal(0, 1, size=(32, 1)).astype(np.float32)
            def func(s):
                h = np.tanh(s @ hidden)
                outv = h @ out
                return np.clip(outv, -1, 1).squeeze(-1)
            return func, "mlp"

    def train_step(self):
        # 1. Sample a reward function
        func, lbl = self.sample_reward_func()
        # 2. Sample K_enc states for encoding
        idx_enc = self.rng.choice(self.n, size=self.K_enc, replace=False)
        states_enc = self.obs[idx_enc]          # (K_enc, state_dim)
        rewards_enc = func(states_enc.numpy())  # (K_enc,)
        states_enc = states_enc.to(self.device)
        rewards_enc = torch.tensor(rewards_enc, dtype=torch.float32,
                                   device=self.device).unsqueeze(-1)

        # 3. Sample K_dec states for decoding
        idx_dec = self.rng.choice(self.n, size=self.K_dec, replace=False)
        states_dec = self.obs[idx_dec]          # (K_dec, state_dim)
        rewards_dec = func(states_dec.numpy())  # (K_dec,)
        states_dec = states_dec.to(self.device)
        rewards_dec = torch.tensor(rewards_dec, dtype=torch.float32,
                                   device=self.device).unsqueeze(-1)

        # 4. Forward through encoder
        z, z_mean, z_logvar = self.enc.encode(states_enc, rewards_enc)

        # 5. Decoder prediction on decoding set
        z_expanded = z.unsqueeze(1).expand(-1, self.K_dec, -1)   # (B=1, K_dec, latent)
        pred = self.dec(states_dec.unsqueeze(0), z.squeeze(0))   # (1, 1)
        pred = pred.squeeze(0)                                  # (1,)

        # 6. Losses
        mse = F.mse_loss(pred, rewards_dec.unsqueeze(0))
        kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        loss = mse + self.beta * kl

        # 7. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), lbl

    def train(self, epochs: int):
        for epoch in range(epochs):
            loss, lbl = self.train_step()
            if (epoch + 1) % 10 == 0:
                print(f"[FRE] Epoch {epoch+1}/{epochs} | loss={loss:.4f} | task={lbl}")