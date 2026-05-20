"""
Training loop for Simformer
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingLoop:
    """
    Training loop for Simformer.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 use_vesde: bool = True):
        """
        Initialize training loop.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            use_vesde: Use VESDE
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_vesde = use_vesde
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
    def train(self, 
              dataloader: torch.utils.data.DataLoader,
              epochs: int = 10,
              save_path: Optional[str] = None,
              validate: bool = False,
              val_dataloader: Optional[torch.utils.data.DataLoader] = None,
            ):
        """
        Train the model.
        
        Args:
            dataloader: Training data loader
            epochs: Number of epochs
            save_path: Path to save model
            validate: Validate during training
            val_dataloader: Validation data loader
        Returns:
            losses: List of losses
        """
        losses = []
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Get batch data
                inputs, targets, masks = batch
                batch_size = inputs.size(0)
                
                # Sample noise level
                t = torch.rand(batch_size, device=inputs.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                scores = self.model(inputs, masks, t)
                
                # Compute loss
                loss = self.criterion(scores, targets)
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Store loss
                epoch_losses.append(loss.item())
                self.step += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Step {self.step}, Loss: {loss.item():.4f}")
            
            # Compute average loss for epoch
            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)
            
            # Validate if needed
            if validate and val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            
            # Save model if improved
            if save_path is not None and epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'step': self.step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss,
                }, f"{save_path}/model_epoch_{epoch}.pt")
            
            # Update scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()
            
            logger.info(f"Epoch {epoch} completed. Loss: {epoch_loss:.4f}")
        
        return losses
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
        Returns:
            loss: Validation loss
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets, masks = batch
            scores = self.model(inputs, masks, torch.zeros(inputs.size(0), device=inputs.device))
            loss = self.criterion(scores, targets)
            val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        self.model.train()
        return val_loss