#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
python -m pip install --quiet --upgrade pip
pip install -r requirements.txt

# Pre‑train on Phase 2 (AppleRetrieval)
echo "Pre‑training Phase 2 (BC) – started."
python - <<'PY'
import torch
from src.envs.apples_retrieval import AppleRetrieval
from src.agents.policy_network import PolicyNetwork
from src.agents.trainer import Trainer

env = AppleRetrieval(phase=2, M=10)
policy = PolicyNetwork(obs_dim=1, action_dim=2)
trainer = Trainer(policy, env, device='cpu',
                  lr=1e-2, gamma=0.99,
                  buffer_size=0,  # no BC during pre‑training
                  bc_coef=0.0,
                  ewc_coef=0.0)
trainer.train(episodes=200, verbose=False)
torch.save(policy.state_dict(), "pretrained.pt")
print("Pre‑training Phase 2 (BC) – finished.")
PY

# Fine‑tune without BC
echo "Fine‑tuning without BC – started."
python - <<'PY'
import torch
from src.envs.apples_retrieval import AppleRetrieval
from src.agents.policy_network import PolicyNetwork
from src.agents.trainer import Trainer
import os

policy = PolicyNetwork(obs_dim=1, action_dim=2)
policy.load_state_dict(torch.load("pretrained.pt"))
env = AppleRetrieval(phase=0, M=10)  # full task
trainer = Trainer(policy, env, device='cpu',
                  lr=1e-2, gamma=0.99,
                  buffer_size=0,  # no BC buffer
                  bc_coef=0.0,
                  ewc_coef=0.0)
trainer.train(episodes=200, verbose=False)
# Evaluate
from src.agents.trainer import evaluate
success_rate = evaluate(policy, env, episodes=50)
print(f"Fine‑tuning without BC – success rate: {success_rate:.2f}")
PY

# Fine‑tune with BC
echo "Fine‑tuning with BC – started."
python - <<'PY'
import torch
from src.envs.apples_retrieval import AppleRetrieval
from src.agents.policy_network import PolicyNetwork
from src.agents.trainer import Trainer
import os

policy = PolicyNetwork(obs_dim=1, action_dim=2)
policy.load_state_dict(torch.load("pretrained.pt"))
env = AppleRetrieval(phase=0, M=10)  # full task

# Build BC buffer from pre‑trained policy
pre_policy = PolicyNetwork(obs_dim=1, action_dim=2)
pre_policy.load_state_dict(torch.load("pretrained.pt"))
buffer = []
for _ in range(200):
    obs = torch.tensor([pre_policy.sample_action(torch.tensor([1.0]))], dtype=torch.float32)
    # Actually we need state-action pairs from Phase 2
    obs_state = torch.tensor([1.0], dtype=torch.float32)  # Phase 2 observation
    action = pre_policy.act(obs_state)
    buffer.append((obs_state, action))

trainer = Trainer(policy, env, device='cpu',
                  lr=1e-2, gamma=0.99,
                  buffer_size=len(buffer),
                  bc_coef=1.0,
                  ewc_coef=0.0,
                  bc_buffer=buffer)
trainer.train(episodes=200, verbose=False)
# Evaluate
from src.agents.trainer import evaluate
success_rate = evaluate(policy, env, episodes=50)
print(f"Fine‑tuning with BC – success rate: {success_rate:.2f}")
PY