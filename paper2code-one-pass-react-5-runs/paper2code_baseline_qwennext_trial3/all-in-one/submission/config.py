# Configuration file for Simformer reproduction

# Model configuration
MODEL_CONFIG = {
    'token_dim': 64,
    'hidden_dim': 256,
    'n_layers': 4,
    'n_heads': 4,
    'dropout': 0.1,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 10,
    'save_interval': 1,
    'model_save_path': 'models/simformer_model.pth',
}

# Inference configuration
INFERENCE_CONFIG = {
    'n_samples': 1000,
    'timesteps': 100,
}

# Data configuration
DATA_CONFIG = {
    'n_vars': 10,
    'n_samples': 1000,
    'noise_std': 0.1,
    'data_path': 'data/simformer_data.csv',
}

# Diffusion configuration
DIFFUSION_CONFIG = {
    'n_timesteps': 100,
    'beta_start': 0.0001,
    'beta_end': 0.02,
}

# Output configuration
OUTPUT_CONFIG = {
    'output_dir': 'output',
    'output_files': {
        'posterior_samples': 'posterior_samples.csv',
        'likelihood_samples': 'likelihood_samples.csv',
        'conditional_samples': 'conditional_samples.csv',
        'interval_constraints': 'interval_constraints.csv',
    }
}

# Reproduction configuration
REPRODUCE_CONFIG = {
    'seed': 42,
    'run_reproduction': True,
}