import sys
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.simulator_wrapper import SimulatorWrapper
from src.data.tokenizer import Tokenizer
from src.arch.attention_mask_builder import build_attention_mask
from src.arch.transformer_score_model import TransformerScoreModel
from src.train.diffusion_trainer import DiffusionTrainer
from src.infra.utils import set_seed, setup_logging


def main(config_path: str) -> None:
    # Load config (assumed to be a Python file or dict; for simplicity, assume it's a dict via import)
    # Since config loading mechanism isn't specified, we assume config_path points to a Python file
    # that defines a config dictionary. We'll use exec for simplicity in this context.
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)
    config = config.get('config', {})

    # Extract config values
    seed = config.get('seed', 42)
    log_dir = config.get('log_dir', './logs')
    batch_size = config.get('batch_size', 32)
    n_params = config.get('n_params', 10)
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Setup logging and seed
    set_seed(seed)
    logger = setup_logging(log_dir)

    # Initialize components
    simulator = SimulatorWrapper()
    tokenizer = Tokenizer()
    model = TransformerScoreModel().to(device)
    trainer = DiffusionTrainer(model, learning_rate, device)

    # Training loop
    for epoch in range(num_epochs):
        # Generate batch of simulated data
        sim_batch = simulator.generate_batch(batch_size, n_params)
        
        # Tokenize simulated outputs
        tokenized_batch = tokenizer.encode(sim_batch)
        
        # Build attention mask from metadata
        mask = build_attention_mask(sim_batch)
        mask = mask.to(device)
        
        # Convert tokens to tensor (assuming tokens are list of dicts with 'input_ids' key)
        # This is a placeholder assumption; actual token structure should match model input
        input_ids = torch.stack([torch.tensor(t['input_ids']) for t in tokenized_batch]).to(device)
        t = torch.randint(0, model.num_timesteps, (input_ids.size(0),), device=device)
        
        # Train one epoch
        loss = trainer.train_epoch(DataLoader(list(zip(input_ids, mask, t)), batch_size=batch_size))
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        # Optional: checkpointing (if checkpoint_dir is specified)
        checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch + 1}.pth")
            logger.info(f"Checkpoint saved at epoch {epoch + 1}")

    logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimFormer model")
    parser.add_argument("config_path", type=str, help="Path to the training configuration file")
    args = parser.parse_args()
    main(args.config_path)