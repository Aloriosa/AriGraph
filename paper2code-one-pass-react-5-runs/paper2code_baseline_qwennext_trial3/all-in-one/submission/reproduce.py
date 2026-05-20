import torch
import numpy as np
import pandas as pd
import os
import json
from simformer import Simformer, DiffusionModel, SimformerInference
from config import MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG, DATA_CONFIG, DIFFUSION_CONFIG, OUTPUT_CONFIG, REPRODUCE_CONFIG

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_synthetic_data(n_samples=1000, n_vars=10, noise_std=0.1):
    """
    Generate synthetic data for reproduction.
    
    This function generates synthetic data that mimics the structure of the problem
    described in the paper. The data includes:
    - Parameters (θ): Random values representing model parameters
    - Data (x): Values generated from the simulator based on parameters
    - Mask: Indicates which variables are observed
    """
    np.random.seed(REPRODUCE_CONFIG['seed'])
    torch.manual_seed(REPRODUCE_CONFIG['seed'])
    
    # Generate parameters (θ) - these could represent model parameters
    # In the paper, these could be function-valued parameters
    theta = np.random.normal(0, 1, (n_samples, n_vars))
    
    # Generate data (x) - these are the observations
    # In the paper, these could be the simulator outputs
    # For simplicity, we'll use a simple linear relationship
    # x = θ * A + noise
    A = np.random.normal(0, 0.5, (n_vars, n_vars))
    x = np.dot(theta, A)
    
    # Add noise
    noise = np.random.normal(0, noise_std, x.shape)
    x = x + noise
    
    # Create mask - indicate which variables are observed
    # In the paper, this could indicate missing data
    mask = np.random.choice([0, 1], size=(n_samples, n_vars))
    
    # Combine into a single dataset
    data = np.concatenate([theta, x], axis=1)
    
    # Save data
    df = pd.DataFrame(data, columns=[f'theta_{i}' for i in range(n_vars)] + [f'x_{i}' for i in range(n_vars)])
    df['mask'] = [mask[i] for i in range(len(mask))]
    
    # Create output directory
    os.makedirs(OUTPUT_CONFIG['output_dir'], exist_ok=True)
    data_path = os.path.join(OUTPUT_CONFIG['output_dir'], 'synthetic_data.csv')
    df.to_csv(data_path, index=False)
    
    return data, mask

def train_model():
    """
    Train the Simformer model.
    """
    print("Training Simformer model...")
    
    # Set seed for reproducibility
    set_seed(REPRODUCE_CONFIG['seed'])
    
    # Generate synthetic data
    data, mask = generate_synthetic_data(
        n_samples=DATA_CONFIG['n_samples'],
        n_vars=DATA_CONFIG['n_vars'],
        noise_std=DATA_CONFIG['noise_std']
    )
    
    # Create model
    simformer = Simformer(
        token_dim=MODEL_CONFIG['token_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        n_layers=MODEL_CONFIG['n_layers'],
        n_heads=MODEL_CONFIG['n_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    # Create diffusion model
    diffusion_model = DiffusionModel(
        simformer,
        n_timesteps=DIFFUSION_CONFIG['n_timesteps'],
        beta_start=DIFFUSION_CONFIG['beta_start'],
        beta_end=DIFFUSION_CONFIG['beta_end']
    )
    
    # Create inference object
    inference = SimformerInference(diffusion_model, device='cpu')
    
    # Train model
    # For reproduction, we'll use a simple training loop
    optimizer = torch.optim.Adam(inference.model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        optimizer.zero_grad()
        
        # Forward pass
        x = torch.tensor(data, dtype=torch.float32)
        t = torch.rand((data.shape[0], 1))
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        # Get predictions
        predictions = inference.model(x, t, mask_tensor)
        
        # Calculate loss
        loss = torch.mean((predictions - x)**2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # Save model
    model_path = TRAIN_CONFIG['model_save_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(inference.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return inference

def run_inference(inference):
    """
    Run inference with the trained model.
    """
    print("Running inference...")
    
    # Load data
    data_path = os.path.join(OUTPUT_CONFIG['output_dir'], 'synthetic_data.csv')
    df = pd.read_csv(data_path)
    
    # Extract data and mask
    data = df.iloc[:, :-1].values
    mask = df['mask'].apply(lambda x: np.array(eval(x)))
    mask = np.array([m for m in mask])
    
    # Run inference
    posterior_samples = inference.infer_posterior(data, n_samples=INFERENCE_CONFIG['n_samples'])
    likelihood_samples = inference.infer_likelihood(data, n_samples=INFERENCE_CONFIG['n_samples'])
    conditional_samples = inference.infer_conditional(data, condition_vars=list(range(data.shape[1])), n_samples=INFERENCE_CONFIG['n_samples'])
    
    # Save results
    os.makedirs(OUTPUT_CONFIG['output_dir'], exist_ok=True)
    
    # Save posterior samples
    posterior_df = pd.DataFrame(posterior_samples.detach().numpy(), columns=[f'param_{i}' for i in range(posterior_samples.shape[1])])
    posterior_path = os.path.join(OUTPUT_CONFIG['output_dir'], OUTPUT_CONFIG['output_files']['posterior_samples'])
    posterior_df.to_csv(posterior_path, index=False)
    print(f"Posterior samples saved to {posterior_path}")
    
    # Save likelihood samples
    likelihood_df = pd.DataFrame(likelihood_samples.detach().numpy(), columns=[f'param_{i}' for i in range(likelihood_samples.shape[1])])
    likelihood_path = os.path.join(OUTPUT_CONFIG['output_dir'], OUTPUT_CONFIG['output_files']['likelihood_samples'])
    likelihood_df.to_csv(likelihood_path, index=False)
    print(f"Likelihood samples saved to {likelihood_path}")
    
    # Save conditional samples
    conditional_df = pd.DataFrame(conditional_samples.detach().numpy(), columns=[f'param_{i}' for i in range(conditional_samples.shape[1])])
    conditional_path = os.path.join(OUTPUT_CONFIG['output_dir'], OUTPUT_CONFIG['output_files']['conditional_samples'])
    conditional_df.to_csv(conditional_path, index=False)
    print(f"Conditional samples saved to {conditional_path}")
    
    # Generate interval constraints
    # Apply interval constraints
    constraints = [
        {
            'type': 'interval',
            'value': [0.1, 0.9]
        }
    ]
    
    interval_samples = inference.infer_with_guidance(
        data, 
        constraints, 
        n_samples=INFERENCE_CONFIG['n_samples']
    )
    
    interval_df = pd.DataFrame(interval_samples.detach().numpy(), columns=[f'param_{i}' for i in range(interval_samples.shape[1])])
    interval_path = os.path.join(OUTPUT_CONFIG['output_dir'], OUTPUT_CONFIG['output_files']['interval_constraints'])
    interval_df.to_csv(interval_path, index=False)
    print(f"Interval constraint samples saved to {interval_path}
    
    return posterior_samples, likelihood_samples, conditional_samples

def main():
    """
    Main function to run reproduction.
    """
    print("Starting Simformer reproduction...")
    
    # Train model
    inference = train_model()
    
    # Run inference
    posterior_samples, likelihood_samples, conditional_samples = run_inference(inference)
    
    print("Reproduction completed successfully!")
    print("Check the 'output/' directory for generated files.")
    
    # Print summary
    print("\nReproduction Summary:")
    print("- Model trained on synthetic data with 10 parameters")
    print("- Generated samples from posterior, likelihood, and conditionals")
    print("- Applied interval constraints using guided diffusion")
    print("- Results saved in 'output/' directory")
    
    return 0

if __name__ == "__main__":
    main()