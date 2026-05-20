# Hyper‑parameters for the whole reproduction pipeline

ENV_NAME = "CartPole-v1"

# ---------- Pre‑training ----------
PRETRAIN_TIMESTEPS = 200_000
PRETRAIN_LOG_FREQ = 10_000
PRETRAIN_SEED = 42

# ---------- Critical‑state extraction ----------
CRIT_STEPS = 100      # episodes to run for extracting critical states
CRIT_TOP_PERCENT = 0.10  # take top 10% states as critical
CRIT_SEED = 123

# ---------- Refinement ----------
REFINE_TIMESTEPS = 200_000
REFINE_LOG_FREQ = 10_000
REFINE_SEED = 7

# Mixed initial state distribution
P_MIX = 0.5          # probability of starting from a critical state

# RND intrinsic reward coefficient
LAMBDA_RND = 0.01

# Optional mask‑network bonus (not used in this simplified version)
ALPHA_MASK = 0.001