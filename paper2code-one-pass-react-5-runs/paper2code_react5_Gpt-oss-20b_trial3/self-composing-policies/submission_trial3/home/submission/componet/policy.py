import math
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from .module import SelfComposingModule


class CompoNetActorCriticSAC(ActorCriticPolicy):
    """
    Custom policy for SAC that uses CompoNet modules.
    The actor outputs a mean vector; log_std is a separate trainable vector.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr: float = 3e-4,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        net_arch=None,
        *args,
        **kwargs,
    ):
        # For SAC we only need the actor part from the base class
        super().__init__(
            observation_space,
            action_space,
            lr,
            features_extractor_class,
            features_extractor_kwargs,
            net_arch,
            *args,
            **kwargs,
        )

        # Replace the Actor with CompoNet
        self.state_dim = self.features_extractor.features_dim
        self.action_dim = action_space.shape[0]
        self.d_model = 256

        self.componet_modules = nn.ModuleList()
        self.current_module = None
        self._freeze_prev = True

        # For SAC we need a separate log_std vector
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def add_module(self, state_dim: int, action_dim: int):
        """Add a new self‑composing module and freeze the previous ones."""
        new_mod = SelfComposingModule(state_dim, action_dim, self.d_model)
        if self.componet_modules:
            for m in self.componet_modules:
                for p in m.parameters():
                    p.requires_grad = False
        self.componet_modules.append(new_mod)
        self.current_module = new_mod

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        # Extract features
        features = self.features_extractor(obs)
        # Gather previous module outputs
        if len(self.componet_modules) == 1:
            prev_outs = torch.empty(obs.shape[0], 0, self.action_dim, device=obs.device)
            action_mean = self.current_module(features, prev_outs)
        else:
            prev_outs_list = []
            for m in self.componet_modules[:-1]:
                out = m(features, torch.empty(obs.shape[0], 0, self.action_dim, device=obs.device))
                prev_outs_list.append(out.unsqueeze(1))
            prev_outs = torch.cat(prev_outs_list, dim=1)  # (B, n_prev, act)
            action_mean = self.current_module(features, prev_outs)

        action_std = torch.exp(self.log_std)
        if deterministic:
            return action_mean
        else:
            return torch.distributions.Normal(action_mean, action_std).rsample()

    @property
    def action_distribution(self):
        # Not used directly
        pass


class CompoNetActorCriticPPO(ActorCriticPolicy):
    """
    Custom policy for PPO that uses CompoNet modules.
    Outputs logits for a categorical distribution (discrete action space).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr: float = 3e-4,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        net_arch=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr,
            features_extractor_class,
            features_extractor_kwargs,
            net_arch,
            *args,
            **kwargs,
        )

        self.state_dim = self.features_extractor.features_dim
        self.action_dim = action_space.n  # Discrete
        self.d_model = 256

        self.componet_modules = nn.ModuleList()
        self.current_module = None

    def add_module(self, state_dim: int, action_dim: int):
        new_mod = SelfComposingModule(state_dim, action_dim, self.d_model)
        if self.componet_modules:
            for m in self.componet_modules:
                for p in m.parameters():
                    p.requires_grad = False
        self.componet_modules.append(new_mod)
        self.current_module = new_mod

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.features_extractor(obs)
        if len(self.componet_modules) == 1:
            prev_outs = torch.empty(obs.shape[0], 0, self.action_dim, device=obs.device)
            logits = self.current_module(features, prev_outs)
        else:
            prev_outs_list = []
            for m in self.componet_modules[:-1]:
                out = m(features, torch.empty(obs.shape[0], 0, self.action_dim, device=obs.device))
                prev_outs_list.append(out.unsqueeze(1))
            prev_outs = torch.cat(prev_outs_list, dim=1)
            logits = self.current_module(features, prev_outs)

        if deterministic:
            action = logits.argmax(-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action, None, logits