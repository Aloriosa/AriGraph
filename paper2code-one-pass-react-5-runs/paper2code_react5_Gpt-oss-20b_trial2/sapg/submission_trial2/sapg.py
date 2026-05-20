import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os, copy

from policy import PolicyNetwork, ValueNetwork
from utils import compute_gae, create_dataloader, set_seed

# ==================================================
# Hyperparameters (can be tuned)
# ==================================================
ENV_NAME = "Pendulum-v1"
NUM_ENV = 256            # total parallel environments
NUM_POLICIES = 3         # one leader + two followers
BLOCK_SIZE = NUM_ENV // NUM_POLICIES
NUM_STEPS = 16           # steps per rollout
GAMMA = 0.99
TAU = 0.95
LR = 5e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.0
VALUE_COEF = 0.5
GRAD_CLIP = 0.5
MINI_EPOCHS = 4
BATCH_SIZE = 2048
TOTAL_EPOCHS = 20        # shortened for demo
EVAL_EPISODES = 50
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# ==================================================
# Environment setup
# ==================================================
env_fns = [lambda: gym.make(ENV_NAME, render_mode=None) for _ in range(NUM_ENV)]
vec_env = gym.vector.SyncVectorEnv(env_fns)

obs_dim = vec_env.single_observation_space.shape[0]
act_dim = vec_env.single_action_space.shape[0]

# ==================================================
# Policies and optimizers
# ==================================================
policies = []
values = []
optimizers = []

for _ in range(NUM_POLICIES):
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value = ValueNetwork(obs_dim).to(device)
    opt = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=LR)
    policies.append(policy)
    values.append(value)
    optimizers.append(opt)

# ==================================================
# Helper: collect rollout for a block
# ==================================================
def collect_rollout(policy, value, env_idx_start, env_idx_end):
    obs, _ = vec_env.reset()
    obs_block = obs[env_idx_start:env_idx_end]

    states, actions, log_probs, rewards, dones, values_pred = [], [], [], [], [], []

    for _ in range(NUM_STEPS):
        obs_t = torch.from_numpy(obs_block).float().to(device)
        with torch.no_grad():
            action, logp = policy.get_action(obs_t)
            val = value(obs_t)
        actions.append(action.cpu().numpy())
        log_probs.append(logp.cpu().numpy())
        values_pred.append(val.cpu().numpy())

        step_action = action.cpu().numpy()
        obs_step, rewards_step, dones_step, _, _ = vec_env.step(step_action)

        # Store transitions
        states.append(obs_block)
        rewards.append(rewards_step[env_idx_start:env_idx_end])
        dones.append(dones_step[env_idx_start:env_idx_end])

        obs_block = obs_step[env_idx_start:env_idx_end]

    # Convert to arrays
    states = np.concatenate(states, axis=0)          # (T*block, obs_dim)
    actions = np.concatenate(actions, axis=0)        # (T*block, act_dim)
    log_probs = np.concatenate(log_probs, axis=0)    # (T*block)
    rewards = np.concatenate(rewards, axis=0)        # (T*block)
    dones = np.concatenate(dones, axis=0)            # (T*block)
    values_pred = np.concatenate(values_pred, axis=0)  # (T*block)

    # Bootstrap value for last state
    with torch.no_grad():
        next_val = value(torch.from_numpy(obs_block).float().to(device)).cpu().numpy()

    advantages, returns = compute_gae(rewards, values_pred, dones, next_val, gamma=GAMMA, tau=TAU)
    return states, actions, log_probs, returns, advantages

# ==================================================
# Main training loop
# ==================================================
for epoch in tqdm(range(TOTAL_EPOCHS), desc="Epochs"):
    # Keep a copy of the current leader for μ calculation
    old_leader_policy = copy.deepcopy(policies[0])

    # Collect data for each policy
    all_data = []
    for i in range(NUM_POLICIES):
        start = i * BLOCK_SIZE
        end = (i + 1) * BLOCK_SIZE
        states, actions, log_probs, returns, advs = collect_rollout(
            policies[i], values[i], start, end
        )
        all_data.append({
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advs": advs,
        })

    # ----- Leader update (policy 0) -----
    leader_idx = 0
    leader = policies[leader_idx]
    leader_val = values[leader_idx]
    leader_opt = optimizers[leader_idx]

    # Own data
    own = all_data[leader_idx]
    own_states = torch.from_numpy(own["states"]).float().to(device)
    own_actions = torch.from_numpy(own["actions"]).float().to(device)
    own_old_logp = torch.from_numpy(own["log_probs"]).float().to(device)
    own_returns = torch.from_numpy(own["returns"]).float().to(device)
    own_advs = torch.from_numpy(own["advs"]).float().to(device)

    # Off-policy data from all followers
    follower_states_list = []
    follower_actions_list = []
    follower_old_logp_list = []
    follower_returns_list = []
    follower_advs_list = []

    for j in range(1, NUM_POLICIES):
        follower = all_data[j]
        follower_states_list.append(follower["states"])
        follower_actions_list.append(follower["actions"])
        follower_old_logp_list.append(follower["log_probs"])
        follower_returns_list.append(follower["returns"])
        follower_advs_list.append(follower["advs"])

    # Concatenate all follower data
    f_states = np.concatenate(follower_states_list, axis=0)
    f_actions = np.concatenate(follower_actions_list, axis=0)
    f_old_logp = np.concatenate(follower_old_logp_list, axis=0)
    f_returns = np.concatenate(follower_returns_list, axis=0)
    f_advs = np.concatenate(follower_advs_list, axis=0)

    # Subsample follower data to match size of own data
    n = own_states.shape[0]
    idx = np.random.choice(f_states.shape[0], n, replace=False)
    f_states = f_states[idx]
    f_actions = f_actions[idx]
    f_old_logp = f_old_logp[idx]
    f_returns = f_returns[idx]
    f_advs = f_advs[idx]

    # Convert follower tensors
    f_states_t = torch.from_numpy(f_states).float().to(device)
    f_actions_t = torch.from_numpy(f_actions).float().to(device)
    f_old_logp_t = torch.from_numpy(f_old_logp).float().to(device)
    f_returns_t = torch.from_numpy(f_returns).float().to(device)
    f_advs_t = torch.from_numpy(f_advs).float().to(device)

    # Compute new log probs under current leader
    with torch.no_grad():
        # For off-policy data: we also need logp_old_leader for μ
        old_leader_mean, old_leader_std = old_leader_policy.forward(f_states_t)
        old_leader_dist = torch.distributions.Normal(old_leader_mean, old_leader_std)
        old_leader_logp_follower = old_leader_dist.log_prob(f_actions_t).sum(dim=-1)

    # Leader new log probs for own and follower data
    leader_mean, leader_std = leader.forward(own_states)
    leader_dist = torch.distributions.Normal(leader_mean, leader_std)
    own_new_logp = leader_dist.log_prob(own_actions).sum(dim=-1)

    leader_mean_f, leader_std_f = leader.forward(f_states_t)
    leader_dist_f = torch.distributions.Normal(leader_mean_f, leader_std_f)
    f_new_logp = leader_dist_f.log_prob(f_actions_t).sum(dim=-1)

    # Compute ratios and clipped surrogate losses
    # On‑policy part
    ratio_own = torch.exp(own_new_logp - own_old_logp)
    surr1_own = ratio_own * own_advs
    surr2_own = torch.clamp(ratio_own, 1 - CLIP_EPS, 1 + CLIP_EPS) * own_advs
    policy_loss_own = -torch.min(surr1_own, surr2_own).mean()

    # Off‑policy part with μ‑scaled clipping
    ratio_f = torch.exp(f_new_logp - f_old_logp_t)
    mu_f = torch.exp(old_leader_logp_follower - f_old_logp_t)
    lower = mu_f * (1 - CLIP_EPS)
    upper = mu_f * (1 + CLIP_EPS)
    surr1_f = ratio_f * f_advs_t
    surr2_f = torch.clamp(ratio_f, lower, upper) * f_advs_t
    policy_loss_off = -torch.min(surr1_f, surr2_f).mean()

    # Value loss (combined)
    val_pred_own = leader_val(own_states)
    val_loss_own = ((val_pred_own - own_returns).pow(2)).mean()

    val_pred_f = leader_val(f_states_t)
    val_loss_f = ((val_pred_f - f_returns_t).pow(2)).mean()

    value_loss = val_loss_own + val_loss_f

    # Total loss (no entropy for leader)
    total_loss = policy_loss_own + policy_loss_off + VALUE_COEF * value_loss

    # Optimisation step
    leader_opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(leader.parameters(), GRAD_CLIP)
    torch.nn.utils.clip_grad_norm_(leader_val.parameters(), GRAD_CLIP)
    leader_opt.step()

    # ----- Followers update (on‑policy only) -----
    for i in range(1, NUM_POLICIES):
        policy = policies[i]
        value = values[i]
        opt = optimizers[i]
        data = all_data[i]
        states_t = torch.from_numpy(data["states"]).float().to(device)
        actions_t = torch.from_numpy(data["actions"]).float().to(device)
        old_logp_t = torch.from_numpy(data["log_probs"]).float().to(device)
        returns_t = torch.from_numpy(data["returns"]).float().to(device)
        advs_t = torch.from_numpy(data["advs"]).float().to(device)

        # New log probs
        mean, std = policy.forward(states_t)
        dist = torch.distributions.Normal(mean, std)
        new_logp = dist.log_prob(actions_t).sum(dim=-1)

        ratio = torch.exp(new_logp - old_logp_t)
        surr1 = ratio * advs_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs_t
        policy_loss = -torch.min(surr1, surr2).mean()

        val_pred = value(states_t)
        value_loss = ((val_pred - returns_t).pow(2)).mean()

        loss = policy_loss + VALUE_COEF * value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(value.parameters(), GRAD_CLIP)
        opt.step()

# ==================================================
# Evaluation of leader policy
# ==================================================
eval_env = gym.make(ENV_NAME, render_mode=None)
total_reward = 0.0
for _ in range(EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action, _ = policies[0].get_action(obs_t, deterministic=True)
        obs, reward, done, _, _ = eval_env.step(action.squeeze(0).cpu().numpy())
        ep_reward += reward
    total_reward += ep_reward
avg_reward = total_reward / EVAL_EPISODES

# Save results
with open("results.txt", "w") as f:
    f.write(f"Average evaluation reward over {EVAL_EPISODES} episodes: {avg_reward:.2f}\\n")
print(f"Average evaluation reward over {EVAL_EPISODES} episodes: {avg_reward:.2f}")