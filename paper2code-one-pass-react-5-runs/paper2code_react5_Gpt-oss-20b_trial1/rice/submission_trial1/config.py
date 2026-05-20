# Configuration parameters for the RICE implementation
ENV_NAME = "Hopper-v3"          # Change to other Mujoco envs if desired
TOTAL_TIMESTEPS = 200_000       # Steps for pre‑training PPO
MASK_TRAIN_TIMESTEPS = 50_000   # Steps for training the mask network
REFINE_TIMESTEPS = 200_000      # Steps for RICE refinement
P = 0.25                       # Probability of resetting to a critical state
LAMBDA = 0.01                  # Weight of RND bonus
ALPHA = 0.0001                  # Bonus for mask network to encourage blinding
SEED = 42                       # Random seed for reproducibility