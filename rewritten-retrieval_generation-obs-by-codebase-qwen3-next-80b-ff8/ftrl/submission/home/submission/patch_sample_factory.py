#!/usr/bin/env python3
"""
Patch SampleFactory to support knowledge retention techniques (BC, EWC, KS)
for fine-tuning with pre-trained models.
This is a minimal patch to add the required functionality without modifying the core library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sample_factory.algo.utils.rl_utils import compute_returns
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.learning.batcher import Batcher
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
import numpy as np
import logging

# Add BC loss calculation
def compute_bc_loss(policy_outputs, expert_policy_outputs, loss_type='kl_divergence'):
    """
    Compute Behavioral Cloning loss between current policy and expert policy.
    """
    if loss_type == 'kl_divergence':
        # KL divergence between policy distributions
        log_prob_diff = policy_outputs['log_prob_actions'] - expert_policy_outputs['log_prob_actions']
        bc_loss = torch.mean(log_prob_diff)
        return bc_loss
    elif loss_type == 'mse':
        # MSE between action probabilities
        action_probs = torch.exp(policy_outputs['log_prob_actions'])
        expert_probs = torch.exp(expert_policy_outputs['log_prob_actions'])
        bc_loss = F.mse_loss(action_probs, expert_probs)
        return bc_loss
    else:
        raise ValueError(f"Unknown BC loss type: {loss_type}")

# Add EWC loss calculation
def compute_ewc_loss(model, fisher_matrix, importance_lambda):
    """
    Compute Elastic Weight Consolidation loss.
    """
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            # Fisher information matrix approximates the curvature of the loss function
            # We penalize changes to parameters that were important for the pre-training task
            fisher = fisher_matrix[name]
            param_diff = param - model.pretrained_params[name]
            ewc_loss += torch.sum(fisher * (param_diff ** 2))
    return importance_lambda * ewc_loss

# Add Kickstarting loss calculation
def compute_ks_loss(policy_outputs, teacher_policy_outputs, temperature=1.0):
    """
    Compute Kickstarting loss using distillation.
    """
    # Use KL divergence between teacher and student policy distributions
    # with temperature scaling for smoother distributions
    student_log_probs = policy_outputs['log_prob_actions'] / temperature
    teacher_probs = torch.exp(teacher_policy_outputs['log_prob_actions'] / temperature)
    
    # Compute KL divergence: KL(P_teacher || P_student)
    ks_loss = torch.sum(teacher_probs * (teacher_policy_outputs['log_prob_actions'] - policy_outputs['log_prob_actions']))
    return ks_loss

# Extend the ActorCritic class to support pre-trained model loading
class FineTunedActorCritic(ActorCritic):
    def __init__(self, obs_space, action_space, cfg):
        super().__init__(obs_space, action_space, cfg)
        self.pretrained_params = {}
        self.fisher_matrix = {}
        self.is_pretrained_loaded = False
        
    def load_pretrained_model(self, model_path):
        """Load pre-trained model weights and store them for EWC/BC/KS"""
        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # Store original parameters for EWC
            for name, param in self.named_parameters():
                self.pretrained_params[name] = param.data.clone()
            self.is_pretrained_loaded = True
            
            # Compute Fisher information matrix (simplified version)
            # In practice, this would be computed during pre-training
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # Use a simple approximation: 1.0 for all parameters
                    self.fisher_matrix[name] = torch.ones_like(param.data)
    
    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        # Call parent forward
        result = super().forward(normalized_obs_dict, rnn_states, values_only)
        return result

# Extend the Learner class to support knowledge retention losses
class KnowledgeRetentionLearner(Learner):
    def __init__(self, cfg, policy, batcher, env_info):
        super().__init__(cfg, policy, batcher, env_info)
        self.bc_loss_weight = getattr(cfg, 'bc_loss_weight', 0.0)
        self.ewc_lambda = getattr(cfg, 'ewc_lambda', 0.0)
        self.ks_loss_weight = getattr(cfg, 'ks_loss_weight', 0.0)
        self.bc_loss_type = getattr(cfg, 'bc_loss_type', 'kl_divergence')
        self.ks_temperature = getattr(cfg, 'ks_temperature', 1.0)
        
    def _compute_loss(self, batch, timing):
        # Call parent method to get base PPO loss
        loss, metrics = super()._compute_loss(batch, timing)
        
        # Add BC loss if enabled
        if self.bc_loss_weight > 0 and self.policy.is_pretrained_loaded:
            # We need to compute the expert policy outputs
            # This is a simplification - in practice we'd have a separate expert policy
            with torch.no_grad():
                expert_outputs = self.policy(batch['obs'], batch['rnn_states'])
            
            bc_loss = compute_bc_loss(
                batch, 
                expert_outputs, 
                loss_type=self.bc_loss_type
            )
            loss += self.bc_loss_weight * bc_loss
            metrics['bc_loss'] = bc_loss.item()
        
        # Add EWC loss if enabled
        if self.ewc_lambda > 0 and self.policy.is_pretrained_loaded:
            ewc_loss = compute_ewc_loss(
                self.policy, 
                self.policy.fisher_matrix, 
                self.ewc_lambda
            )
            loss += ewc_loss
            metrics['ewc_loss'] = ewc_loss.item()
        
        # Add KS loss if enabled
        if self.ks_loss_weight > 0 and self.policy.is_pretrained_loaded:
            with torch.no_grad():
                teacher_outputs = self.policy(batch['obs'], batch['rnn_states'])
            
            ks_loss = compute_ks_loss(
                batch, 
                teacher_outputs, 
                temperature=self.ks_temperature
            )
            loss += self.ks_loss_weight * ks_loss
            metrics['ks_loss'] = ks_loss.item()
        
        return loss, metrics

# Extend the Batcher class to handle knowledge retention
class KnowledgeRetentionBatcher(Batcher):
    def __init__(self, evt_loop, policy_id, buffer_mgr, cfg, env_info):
        super().__init__(evt_loop, policy_id, buffer_mgr, cfg, env_info)
        
        # Add knowledge retention specific configuration
        self.bc_loss_weight = getattr(cfg, 'bc_loss_weight', 0.0)
        self.ewc_lambda = getattr(cfg, 'ewc_lambda', 0.0)
        self.ks_loss_weight = getattr(cfg, 'ks_loss_weight', 0.0)

# Register the custom components
def register_custom_components():
    """Register custom components with SampleFactory"""
    from sample_factory.algo.utils.misc import register_custom_model
    
    # Register the fine-tuned actor-critic model
    register_custom_model('fine_tuned_actor_critic', FineTunedActorCritic)
    
    # Register the knowledge retention learner
    from sample_factory.algo.learning.learner import register_custom_learner
    register_custom_learner('knowledge_retention_learner', KnowledgeRetentionLearner)
    
    # Register the knowledge retention batcher
    from sample_factory.algo.learning.batcher import register_custom_batcher
    register_custom_batcher('knowledge_retention_batcher', KnowledgeRetentionBatcher)

# Call registration function
register_custom_components()

# Add command line arguments for knowledge retention
def add_knowledge_retention_args(parser):
    """Add knowledge retention specific arguments to the parser"""
    parser.add_argument('--bc_loss_weight', type=float, default=0.0, help='Weight for behavioral cloning loss')
    parser.add_argument('--bc_loss_type', type=str, default='kl_divergence', choices=['kl_divergence', 'mse'], help='Type of BC loss')
    parser.add_argument('--ewc_lambda', type=float, default=0.0, help='EWC regularization strength')
    parser.add_argument('--ewc_fisher_samples', type=int, default=100, help='Number of samples for Fisher matrix estimation')
    parser.add_argument('--ks_loss_weight', type=float, default=0.0, help='Weight for kickstarting loss')
    parser.add_argument('--ks_temperature', type=float, default=1.0, help='Temperature for knowledge distillation')
    return parser

# Add the arguments to SampleFactory's argument parser
from sample_factory.cfg.arguments import parse_sf_args
original_parse_sf_args = parse_sf_args

def patched_parse_sf_args(argv=None, evaluation=False):
    parser, args = original_parse_sf_args(argv, evaluation)
    parser = add_knowledge_retention_args(parser)
    args, _ = parser.parse_known_args(argv)
    return parser, args

# Patch the parse_sf_args function
parse_sf_args = patched_parse_sf_args