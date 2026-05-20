#!/usr/bin/env python3
"""
CompoNet implementation for continual reinforcement learning.
Implements the growable modular neural network with attention-based policy composition.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

# Import utility functions
from utils import Identity, get_position_encoding, logit2prob

class CompoNetAgent(nn.Module):
    """
    CompoNet agent that combines multiple policy modules using attention mechanisms.
    This implementation follows the paper's description of a growable modular neural network
    that avoids catastrophic forgetting and enables knowledge transfer.
    """
    def __init__(self, obs_dim, act_dim, prev_paths: List[str] = [], map_location=None):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Initialize the network for logstd (shared across all modules)
        self.net_logstd = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        
        # Initialize the CompoNet for mean action prediction
        prev_units = []
        
        # Load previous modules if provided
        if len(prev_paths) > 0:
            # Load the first module with special wrapper
            first_module = torch.load(f"{prev_paths[0]}/net_mean.pt", map_location=map_location)
            prev_units.append(FirstModuleWrapper(
                model=first_module,
                ret_probs=False,
                transform_output=take_first
            ))
            
            # Load remaining modules
            for p in prev_paths[1:]:
                module = torch.load(f"{p}/net_mean.pt", map_location=map_location)
                prev_units.append(module)
        
        # Create internal policy network
        # Input: obs_dim + hidden_dim (256), Output: act_dim
        internal_policy = nn.Sequential(
            nn.Linear(obs_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        
        # Initialize CompoNet with previous modules
        self.net_mean = CompoNet(
            previous_units=prev_units,
            input_dim=obs_dim,
            hidden_dim=256,
            out_dim=act_dim,
            internal_policy=internal_policy,
            ret_probs=False,
            encoder=None  # No encoder for state-based observations
        )
    
    def forward(self, x, writer=None, global_step=None):
        """
        Forward pass of the CompoNet agent.
        Returns mean and logstd for the action distribution.
        """
        if writer is None or global_step is None:
            # Normal forward pass
            mean = self.net_mean(x)[0]
        else:
            # Forward pass with attention visualization
            mean, _phi, att_in, att_out, int_pol, head_out = self.net_mean(
                x, return_atts=True, ret_int_pol=True, ret_head_out=True
            )
            
            # Log attention values to tensorboard
            for i, v in enumerate(att_in.mean(0)[0].detach()):
                writer.add_scalar(f"charts/att_in_{i}", v.item(), global_step)
            for i, v in enumerate(att_out.mean(0)[0].detach()):
                writer.add_scalar(f"charts/att_out_{i}", v.item(), global_step)
            
            # Log distances between components
            with torch.no_grad():
                dist_out_int_pol = (mean - int_pol).abs().mean().item()
                dist_out_head_out = (mean - head_out).abs().mean().item()
                dist_int_pol_head_out = (int_pol - head_out).abs().mean().item()
                
                writer.add_scalar(f"charts/dist_out_int_pol", dist_out_int_pol, global_step)
                writer.add_scalar(f"charts/dist_out_head_out", dist_out_head_out, global_step)
                writer.add_scalar(f"charts/dist_int_pol_head_out", dist_int_pol_head_out, global_step)
        
        # Compute logstd
        logstd = self.net_logstd(x)
        return mean, logstd
    
    def save(self, dirname):
        """
        Save the CompoNet agent components.
        """
        os.makedirs(dirname, exist_ok=True)
        
        # Remove previous_units to avoid serialization issues
        del self.net_mean.previous_units
        
        # Save components separately
        torch.save(self.net_mean, f"{dirname}/net_mean.pt")
        torch.save(self.net_logstd, f"{dirname}/net_logstd.pt")
    
    @staticmethod
    def load(dirname, obs_dim, act_dim, prev_paths: List[str] = [], map_location=None):
        """
        Load a CompoNet agent from saved components.
        """
        print("Loading previous:", prev_paths)
        
        # Create new agent
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=prev_paths,
            map_location=map_location
        )
        
        # Load logstd
        model.net_logstd = torch.load(
            f"{dirname}/net_logstd.pt", map_location=map_location
        )
        
        # Load net_mean
        net_mean = torch.load(f"{dirname}/net_mean.pt", map_location=map_location)
        
        # Copy state dict
        curr = model.net_mean.state_dict()
        other = net_mean.state_dict()
        for k in other:
            curr[k] = other[k]
        model.net_mean.load_state_dict(curr)
        
        return model


class CompoNet(nn.Module):
    """
    The core CompoNet module that implements the growable modular neural network.
    Uses attention mechanisms to combine previous policies and learn new ones.
    """
    def __init__(
        self,
        previous_units: List[nn.Module],
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        internal_policy: nn.Module,
        ret_probs: bool,
        encoder: Optional[nn.Module] = None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        """
        Initialize the CompoNet module.
        
        Args:
            previous_units: List of previously trained policy modules
            input_dim: Dimension of input state
            hidden_dim: Hidden dimension for attention mechanisms
            out_dim: Dimension of output (action space)
            internal_policy: Network that combines attention output with state
            ret_probs: Whether to return probabilities or logits
            encoder: Optional encoder for observations
            device: Device to run on
            proj_bias: Whether to use bias in linear projections
            att_heads_init: Initialization for attention heads
        """
        super(CompoNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.ret_probs = ret_probs
        self.internal_policy = internal_policy
        self.encoder = encoder if encoder is not None else Identity()
        self.att_temp = np.sqrt(hidden_dim)  # Attention temperature
        
        # This attribute distinguishes between current and previous modules
        self.is_prev = False
        
        # Linear transformations for output attention head
        self.headout_wq = att_heads_init(nn.Linear(input_dim, hidden_dim, bias=proj_bias))
        self.headout_wk = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))
        
        # Linear transformations for input attention head
        self.headin_wq = att_heads_init(nn.Linear(input_dim, hidden_dim, bias=proj_bias))
        self.headin_wk = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))
        self.headin_wv = att_heads_init(nn.Linear(out_dim, hidden_dim, bias=proj_bias))
        
        # Pre-compute positional encodings for input attention head
        n_prev = len(previous_units)
        pe1 = torch.tensor(
            get_position_encoding(seq_len=n_prev + 1, d=out_dim),
            dtype=torch.float32,
            device=device,
        )  # (n_prev+1, out_dim)
        self.pe1 = pe1[None, :, :]  # (1, n_prev+1, out_dim)
        
        # Pre-compute positional encodings for output attention head (if needed)
        if n_prev >= 2:
            self.pe0 = self.pe1[:, :-1, :]  # (1, n_prev, out_dim)
        else:
            self.pe0 = None
        
        # Prepare previous units
        for unit in previous_units:
            # Remove previous_units attribute to avoid circular references
            if hasattr(unit, "previous_units"):
                del unit.previous_units
            unit.is_prev = True
            unit.eval()
            # Freeze all parameters of previous modules
            for param in unit.parameters():
                param.requires_grad = False
        
        # Join all previous units into a single sequential model
        self.previous_units = nn.Sequential(*previous_units)
    
    def _forward_headout(self, s, phi):
        """
        Compute the output attention head.
        Returns the result of the attention head and the employed attention weights.
        
        Args:
            s: Current state representation
            phi: Matrix with results of previous modules
            
        Returns:
            att_dot_val: Attention-weighted combination of previous policies
            att: Attention weights
        """
        # Compute query, keys and values
        query = self.headout_wq(s)
        # Add positional encoding and compute K transformation
        keys = self.headout_wk(phi + self.pe0 if self.pe0 is not None else phi)
        values = phi
        
        # Compute attention weights
        w = torch.matmul(
            query[:, None, :],  # (batch, 1, hidden_dim)
            keys.permute(0, 2, 1),  # (batch, hidden_dim, num_policies)
        )
        
        # Get attention weights with temperature scaling
        att = F.softmax(w / self.att_temp, dim=-1)
        
        # Compute attention-weighted combination
        att_dot_val = torch.matmul(att, values)  # (batch_size, 1, out_dim)
        
        return att_dot_val, att
    
    def _get_internal_policy(self, s, phi):
        """
        Compute the input attention head and the internal policy.
        Returns the result of the internal policy and the employed attention weights.
        
        Args:
            s: Current state representation
            phi: Matrix with results of previous modules and output attention head
            
        Returns:
            policy_out: Output of the internal policy
            att: Attention weights
        """
        # Obtain elements of dot-product attention
        query = self.headin_wq(s)
        values = self.headin_wv(phi)
        keys = self.headin_wk(phi + self.pe1)
        
        # Compute attention weights
        w = torch.matmul(
            query[:, None, :],  # (batch, 1, hidden_dim)
            keys.permute(0, 2, 1),  # (batch, hidden_dim, num_policies)
        )
        
        # Get attention weights with temperature scaling
        att = F.softmax(w / self.att_temp, dim=-1)
        
        # Compute attention-weighted combination
        att_dot_val = torch.matmul(att, values)  # (batch_size, 1, out_dim)
        att_dot_val = att_dot_val[:, 0, :]  # Remove extra dimension: (batch, out_dim)
        
        # Concatenate current state and attention output for internal policy
        policy_in = torch.hstack([att_dot_val, s])  # (batch, out_dim + input_dim)
        
        # Pass through internal policy
        policy_out = self.internal_policy(policy_in)
        
        return policy_out, att
    
    def forward(
        self,
        s,
        ret_encoder_out=False,
        return_atts=False,
        ret_int_pol=False,
        ret_head_out=False,
        prevs_to_noise=0,
    ):
        """
        Forward pass of the CompoNet unit.
        
        This method has two behaviors depending on whether the module is the last module of
        CompoNet or not.
        
        If it is not the last one, the method takes a matrix with the outputs of the preceding
        modules and the current state as input, and returns the same tuple but with the output
        of the module appended to the input matrix. In this case, all keyword arguments are
        ignored. This mode of operation is only intended to be used internally by CompoNet.
        
        If the module is the last one (the one operating in the current task), the method takes
        the current state as the input, and runs the whole CompoNet network to get the final result
        of the model.
        
        Args:
            s: Input state
            ret_encoder_out: Return encoder output
            return_atts: Return attention weights
            ret_int_pol: Return internal policy output
            ret_head_out: Return output of output attention head
            prevs_to_noise: Number of previous modules to replace with noise (for ablation)
            
        Returns:
            pi: Output vector of the CompoNet model
            phi: Matrix of previous policies with current one stacked as last row
            (optionals): encoder output, attention weights, internal policy output, etc.
        """
        # Obtain the outputs of preceding modules (the phi matrix)
        if not self.is_prev:  # If it's the last module
            with torch.no_grad():
                # Get the output of the previous modules in the Phi matrix
                phi, _s = self.previous_units(s)
                
                # Ablation: Replace output of first prevs_to_noise modules with noise
                if prevs_to_noise > 0:
                    if self.ret_probs:
                        # Sample from uniform Dirichlet for probability vectors
                        m = torch.distributions.Dirichlet(
                            torch.tensor([1 / self.out_dim] * self.out_dim)
                        )
                        r = m.sample(sample_shape=[phi.size(0), prevs_to_noise])
                    else:
                        # Sample from normal distribution for logits
                        r = torch.randn((phi.size(0), prevs_to_noise, phi.size(-1)))
                    phi[:, :prevs_to_noise, :] = r
        else:
            # Input to a previous unit must be a tuple (phi, s)
            assert type(s) == tuple, "Input to a previous unit must be a tuple (phi, s)"
            phi, s = s  # phi: (batch, num prev, out dim), s: (batch, input dim)
        
        # Encode input state
        hs = self.encoder(s)
        
        # Compute the result of the output attention head
        out_head, att_head_out = self._forward_headout(hs, phi)
        
        # Get the output of the internal policy
        int_pol_phi = torch.cat([phi, out_head], dim=1)  # Concatenate in num_prev dimension
        logits, att_head_in = self._get_internal_policy(hs, int_pol_phi)
        
        # Compute the final output of the module
        out_head = out_head[:, 0, :]  # (batch, 1, out_dim) -> (batch, out_dim)
        out = out_head + logits
        
        # Normalize output if necessary
        if self.ret_probs:
            out = logit2prob(out)
        
        # Add the resulting policy to the phi matrix
        out = out[:, None, :]  # out: (batch, 1, out_dim)
        phi = torch.cat([phi, out], dim=1)  # Concatenate in num_prev dimension
        
        if self.is_prev:
            return phi, s
        
        out = out[:, 0, :]  # (batch, out_dim)
        
        # Build return value depending on selected options
        ret_vals = [out, phi]
        if ret_encoder_out:
            ret_vals.append(hs)
        if return_atts:
            ret_vals += [att_head_in, att_head_out]
        if ret_int_pol:
            ret_vals.append(logits)
        if ret_head_out:
            ret_vals.append(out_head)
        
        return ret_vals


class FirstModuleWrapper(nn.Module):
    """
    Wrapper for the first module in CompoNet to handle different interfaces.
    """
    def __init__(self, model, ret_probs=False, transform_output=None):
        super().__init__()
        self.model = model
        self.ret_probs = ret_probs
        self.transform_output = transform_output
        
    def forward(self, s):
        # Get output from underlying model
        output = self.model(s)
        
        # Apply transformation if specified
        if self.transform_output is not None:
            output = self.transform_output(output)
        
        # Convert to probability if needed
        if self.ret_probs:
            output = logit2prob(output)
        
        # Return as phi matrix (batch, 1, out_dim)
        return output.unsqueeze(1), s


def take_first(x):
    """
    Helper function to take the first element of a tuple.
    """
    return x[0] if isinstance(x, tuple) else x