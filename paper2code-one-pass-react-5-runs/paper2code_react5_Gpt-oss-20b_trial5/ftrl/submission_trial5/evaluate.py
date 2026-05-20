#!/usr/bin/env python3
"""
Evaluate a policy on AppleRetrieval and report average return.
"""

import torch
import numpy as np
import argparse
import os
from apple_retrieval import AppleRetrieval
from policy import Policy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="finetune/finetune.pt",
                    help="Path to the policy checkpoint")
parser.add_argument("--episodes", type=int, default=200,
                    help="Number of evaluation episodes")
args = parser.parse_args()

# load policy
ckpt = torch.load(args.model, map_location="cpu")
policy = Policy()
policy.load_state_dict(ckpt["policy_state_dict"])
policy.eval()

# evaluation
M = 30
c = 1.0
max_steps = 100
seed = 12345

returns = []
for ep in range(args.episodes):
    env = AppleRetrieval(M=M, c=c, max_steps=max_steps, seed=seed + ep)
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        action = policy.get_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ep_ret += reward
    returns.append(ep_ret)

avg_ret = np.mean(returns)
print(f"Evaluation over {args.episodes} episodes: average return = {avg_ret:.2f}")