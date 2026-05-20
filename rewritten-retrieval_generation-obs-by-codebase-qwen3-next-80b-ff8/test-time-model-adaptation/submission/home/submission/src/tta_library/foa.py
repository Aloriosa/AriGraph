import torch
import torch.nn as nn
import numpy as np
import cma
from torch.nn import functional as F
from models.vpt import PromptViT
from utils.cli_utils import entropy_loss

class FOA(nn.Module):
    """test-time Forward Optimization Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, model:PromptViT, fitness_lambda=0.4):
        super().__init__()
        self.fitness_lambda = fitness_lambda
        self.model = model
        self.es = self._init_cma() # initialization for CMA-ES
        self.best_prompts = model.prompts
        self.best_loss = np.inf
        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)
        self.train_info = None
        self.imagenet_mask = None
        self.entropy_loss = entropy_loss()
        
        # Initialize with zero mean and unit variance for prompts
        self.model.prompts.data = torch.zeros_like(self.model.prompts.data)

    def _init_cma(self):
        """CMA-ES initialization"""
        dim = self.model.prompts.numel()
        popsize = 27 # which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
            'tolx': 1e-6,
            'tolfun': 1e-6,
            'tolfunhist': 1e-6,
            'maxfevals': 1000,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None or self.train_info is None:
            return None
        else:
            # Extract the last 768 dimensions (CLS token features)
            source_mean = self.train_info[1][-768:]  # mean from training
            test_mean = self.hist_stat[-768:]        # current test mean
            return source_mean - test_mean

    def forward(self, x):
        """Main forward pass for FOA adaptation"""
        # Get current activation statistics
        with torch.no_grad():
            features = self.model.layers_cls_features_with_prompts(x)
            batch_mean = torch.mean(features, dim=0)
            
        # Calculate shift vector
        shift_vector = self._get_shift_vector()
        
        # Initialize best values
        self.best_loss = np.inf
        self.best_outputs = None
        batch_means = []
        
        # Sample from CMA-ES and evaluate new solutions
        # Include current best solution in the sample
        prompts, losses = self.es.ask() + [self.best_prompts.flatten().cpu().numpy()], []
        
        for j, prompt in enumerate(prompts):
            # Update model prompts with candidate
            prompt_tensor = torch.tensor(prompt, dtype=torch.float).reshape_as(self.model.prompts).cuda()
            self.model.prompts.data = prompt_tensor
            
            # Forward pass and compute loss
            outputs = self.model(x)
            loss, batch_mean = self._compute_fitness(outputs, features, shift_vector)
            
            # Track batch means for updating historical statistics
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            
            # Update best solution if this is better
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_prompts.data = prompt_tensor.detach().clone()
                self.best_outputs = outputs.detach().clone()
            
            losses.append(loss.item())
            
        # Update CMA-ES with evaluated solutions
        self.es.tell(prompts, losses)
        
        # Update historical statistics
        if len(batch_means) > 0:
            batch_means = torch.cat(batch_means, dim=0).mean(0)
            self._update_hist(batch_means)
        
        # Return best output
        return self.best_outputs
    
    def _compute_fitness(self, outputs, features, shift_vector):
        """Compute fitness function as defined in Eqn. (5)"""
        # Entropy term
        entropy_term = self.entropy_loss(outputs)
        
        # Distribution alignment term (activation discrepancy)
        activation_discrepancy_term = 0
        if shift_vector is not None and self.train_info is not None:
            # Extract the last 768 dimensions (CLS token features)
            test_mean = torch.mean(features, dim=0)[-768:]
            source_mean = self.train_info[1][-768:]
            
            # L2 distance between test and source activation means
            activation_discrepancy = torch.norm(test_mean - source_mean, p=2)
            activation_discrepancy_term = activation_discrepancy
            
        # Combined fitness function (Eqn. 5)
        # Note: We use fitness = - (entropy + lambda * discrepancy) 
        # because CMA-ES minimizes, but we want to maximize fitness
        fitness = entropy_term + self.fitness_lambda * activation_discrepancy_term
        
        return fitness, torch.mean(features, dim=0)
    
    def obtain_origin_stat(self, train_loader):
        """Calculate source domain statistics from training data"""
        print('===> begin calculating mean and variance')
        features = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images)
                features.append(feature)
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0)
        del features
        print('===> calculating mean and variance end')
    
    def reset(self):
        """Reset CMA-ES and historical statistics"""
        self.es = self._init_cma()
        self.hist_stat = None
        self.best_prompts = self.model.prompts
        self.best_loss = np.inf