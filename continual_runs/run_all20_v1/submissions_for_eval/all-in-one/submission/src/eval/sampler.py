import torch
from src.arch.transformer_score_model import TransformerScoreModel
from src.arch.attention_mask_builder import build_attention_mask
from src.data.tokenizer import Tokenizer

class Sampler:
    def __init__(self, score_model: TransformerScoreModel, tokenizer: Tokenizer, num_steps: int = 1000, eta: float = 0.0):
        self.score_model = score_model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.eta = eta  # controls noise level in reverse diffusion (eta=0 -> deterministic)
        self.device = next(score_model.parameters()).device

    def sample_posterior(self, observed_tokens: list[dict], n_samples: int) -> list[dict]:
        """
        Sample from the posterior distribution p(x_unobserved | x_observed) using reverse diffusion.
        observed_tokens: list of token dicts representing observed variables
        n_samples: number of samples to generate
        """
        # Encode observed tokens into tensor format
        batch_size = n_samples
        max_len = max(len(tok['ids']) for tok in observed_tokens) if observed_tokens else 1
        
        # Initialize unobserved tokens as noise
        # We'll create a mask indicating which positions are observed vs unobserved
        observed_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        # Fill in observed tokens (replicate for all samples)
        for i in range(batch_size):
            if observed_tokens:
                tok = observed_tokens[0]  # assume all observed tokens are identical across samples
                seq_len = len(tok['ids'])
                token_ids[i, :seq_len] = torch.tensor(tok['ids'], device=self.device)
                observed_mask[i, :seq_len] = True
        
        # Build attention mask for the observed variables
        sim_metadata = {
            'observed_mask': observed_mask,
            'batch_size': batch_size,
            'seq_len': max_len
        }
        attention_mask = build_attention_mask(sim_metadata)
        
        # Reverse diffusion process
        x_t = torch.randn_like(token_ids, dtype=torch.float32) * 0.1  # start with noise
        x_t = x_t * observed_mask.float() + token_ids.float() * (1 - observed_mask.float())  # fix observed
        
        # Time steps for reverse diffusion
        timesteps = torch.linspace(1, 0, self.num_steps + 1, device=self.device)[1:]
        
        for t in timesteps:
            # Convert t to tensor
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            # Get score prediction
            with torch.no_grad():
                score = self.score_model(x_t.long(), attention_mask, t_tensor)
            
            # Reverse diffusion step: x_{t-1} = x_t + score * dt
            dt = 1.0 / self.num_steps
            noise_scale = self.eta * torch.sqrt(dt)
            noise = torch.randn_like(x_t) * noise_scale
            
            # Update: x_{t-1} = x_t + score * dt + noise
            x_t = x_t + score * dt + noise
            
            # Clamp observed tokens to their original values
            x_t = x_t * (1 - observed_mask.float()) + token_ids.float() * observed_mask.float()
        
        # Convert final samples back to token dicts
        samples = []
        for i in range(batch_size):
            # Convert tensor to list of token ids
            seq = x_t[i].long().tolist()
            # Truncate padding
            seq = [id for id in seq if id != 0]  # assuming 0 is padding
            sample_dict = {'ids': seq}
            samples.append(sample_dict)
        
        # Decode to final output format
        decoded_samples = [self.tokenizer.decode(sample) for sample in samples]
        return decoded_samples

    def sample_likelihood(self, parameter_tokens: list[dict], n_samples: int) -> list[dict]:
        """
        Sample from the likelihood distribution p(y | x) where x are parameters.
        This is equivalent to sampling from the joint distribution conditioned on parameters.
        parameter_tokens: list of token dicts representing parameter variables
        n_samples: number of samples to generate
        """
        batch_size = n_samples
        max_len = max(len(tok['ids']) for tok in parameter_tokens) if parameter_tokens else 1
        
        # Initialize with parameter tokens as fixed
        parameter_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        # Fill in parameter tokens
        for i in range(batch_size):
            if parameter_tokens:
                tok = parameter_tokens[0]  # assume all parameter tokens are identical
                seq_len = len(tok['ids'])
                token_ids[i, :seq_len] = torch.tensor(tok['ids'], device=self.device)
                parameter_mask[i, :seq_len] = True
        
        # Build attention mask for parameters
        sim_metadata = {
            'observed_mask': parameter_mask,
            'batch_size': batch_size,
            'seq_len': max_len
        }
        attention_mask = build_attention_mask(sim_metadata)
        
        # Reverse diffusion: start from noise, condition on parameters
        x_t = torch.randn_like(token_ids, dtype=torch.float32) * 0.1
        x_t = x_t * (1 - parameter_mask.float()) + token_ids.float() * parameter_mask.float()  # fix parameters
        
        timesteps = torch.linspace(1, 0, self.num_steps + 1, device=self.device)[1:]
        
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            with torch.no_grad():
                score = self.score_model(x_t.long(), attention_mask, t_tensor)
            
            dt = 1.0 / self.num_steps
            noise_scale = self.eta * torch.sqrt(dt)
            noise = torch.randn_like(x_t) * noise_scale
            
            x_t = x_t + score * dt + noise
            x_t = x_t * (1 - parameter_mask.float()) + token_ids.float() * parameter_mask.float()  # keep parameters fixed
        
        # Convert to token dicts
        samples = []
        for i in range(batch_size):
            seq = x_t[i].long().tolist()
            seq = [id for id in seq if id != 0]  # remove padding
            sample_dict = {'ids': seq}
            samples.append(sample_dict)
        
        decoded_samples = [self.tokenizer.decode(sample) for sample in samples]
        return decoded_samples