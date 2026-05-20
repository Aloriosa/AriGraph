import sys
import json
import torch
from typing import Dict, List
import logging

from src.data.simulator_wrapper import SimulatorWrapper
from src.data.tokenizer import Tokenizer
from src.arch.attention_mask_builder import build_attention_mask
from src.arch.transformer_score_model import TransformerScoreModel
from src.eval.sampler import sample_posterior, sample_likelihood
from src.infra.utils import set_seed, setup_logging

def main(model_path: str, config_path: str, condition_spec: dict) -> dict:
    """
    Main entry point for evaluation. Loads trained model, samples posteriors/likelihoods
    under arbitrary conditions, and computes metrics against baselines.
    Supports interactive conditioning.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Setup logging
    log_dir = config.get('log_dir', './logs')
    logger = setup_logging(log_dir)
    
    # Initialize components
    simulator = SimulatorWrapper()
    tokenizer = Tokenizer()
    
    # Load trained model
    model = TransformerScoreModel(**config['model_args'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract sampling parameters
    n_samples = config.get('n_samples', 1000)
    batch_size = config.get('batch_size', 100)
    
    results = {}
    
    # Handle posterior sampling under conditioning
    if 'observed' in condition_spec:
        observed_sim = simulator.generate_batch(1, config['n_params'])
        for key, value in condition_spec['observed'].items():
            observed_sim[key] = value
        observed_tokens = tokenizer.encode(observed_sim)
        
        posterior_samples = sample_posterior(observed_tokens, n_samples)
        results['posterior_samples'] = posterior_samples
        
        # Decode samples for analysis
        decoded_posterior = tokenizer.decode(posterior_samples)
        results['decoded_posterior'] = decoded_posterior
    
    # Handle likelihood sampling under parameter conditioning
    if 'parameters' in condition_spec:
        parameter_sim = simulator.generate_batch(1, config['n_params'])
        for key, value in condition_spec['parameters'].items():
            parameter_sim[key] = value
        parameter_tokens = tokenizer.encode(parameter_sim)
        
        likelihood_samples = sample_likelihood(parameter_tokens, n_samples)
        results['likelihood_samples'] = likelihood_samples
        
        # Decode samples for analysis
        decoded_likelihood = tokenizer.decode(likelihood_samples)
        results['decoded_likelihood'] = decoded_likelihood
    
    # Handle arbitrary conditioning (interactive applications)
    if 'conditional' in condition_spec:
        conditional_sim = simulator.generate_batch(1, config['n_params'])
        for key, value in condition_spec['conditional'].items():
            conditional_sim[key] = value
        conditional_tokens = tokenizer.encode(conditional_sim)
        
        # Sample from joint distribution under condition
        conditional_samples = sample_posterior(conditional_tokens, n_samples)
        results['conditional_samples'] = conditional_samples
        decoded_conditional = tokenizer.decode(conditional_samples)
        results['decoded_conditional'] = decoded_conditional
    
    # Compute metrics against baselines (if specified)
    if 'baselines' in config:
        metrics = {}
        for baseline_name, baseline_config in config['baselines'].items():
            if baseline_name == 'true_posterior':
                # Placeholder: in real implementation, compare against analytical or MCMC truth
                pass
            elif baseline_name == 'sbi':
                # Placeholder: compare against SBI-style methods
                pass
            metrics[baseline_name] = {}  # Placeholder for actual metric computation
        results['metrics'] = metrics
    
    # Log results
    logger.info(f"Eval completed. Generated {n_samples} samples for posterior/likelihood.")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/entry/eval.py <model_path> <config_path> <condition_spec_json>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    condition_spec = json.loads(sys.argv[3])
    
    results = main(model_path, config_path, condition_spec)
    print(json.dumps(results, indent=2))