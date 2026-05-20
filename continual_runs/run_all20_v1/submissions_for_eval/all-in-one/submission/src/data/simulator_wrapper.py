import numpy as np
import random
from typing import Dict, Any, List, Optional

class SimulatorWrapper:
    def generate_batch(self, batch_size: int, n_params: int) -> dict:
        """
        Generate a batch of parameter-data pairs from ecological, epidemiological, or neuroscience simulators.
        Each sample includes irregularly sampled time series, missing observations, and function-valued parameters.
        Output is structured as tokenizable sequences with metadata.
        
        Args:
            batch_size: Number of samples to generate
            n_params: Number of parameters per sample
            
        Returns:
            Dictionary with keys:
                - 'parameters': (batch_size, n_params) array of scalar parameters
                - 'data': list of length batch_size, each element is a dict with:
                    - 'time': (T,) array of observation times (irregular)
                    - 'values': (T, D) array of observed values (D = dimensionality)
                    - 'mask': (T, D) boolean array indicating observed (True) vs missing (False)
                - 'metadata': list of dicts with simulation metadata per sample
        """
        batch_parameters = np.random.uniform(0.1, 10.0, size=(batch_size, n_params))
        batch_data = []
        batch_metadata = []
        
        for _ in range(batch_size):
            # Randomly choose simulator type (ecological, epidemiological, neuroscience)
            sim_type = random.choice(['ecological', 'epidemiological', 'neuroscience'])
            
            # Generate irregular time grid (between 50 and 200 time points)
            n_obs = random.randint(50, 200)
            time_points = np.sort(np.random.uniform(0, 100, size=n_obs))
            
            # Generate function-valued parameters (each parameter is a function of time)
            # We represent these as polynomial coefficients modulated by time
            param_functions = []
            for p in range(n_params):
                # Random polynomial of degree 0-3
                degree = random.randint(0, 3)
                coeffs = np.random.uniform(-1, 1, size=degree + 1)
                param_func = lambda t, c=coeffs: np.polyval(c, t)
                param_functions.append(param_func)
            
            # Simulate data based on parameter functions and simulator type
            dim = 1 if sim_type == 'ecological' else (2 if sim_type == 'epidemiological' else 3)
            values = np.zeros((n_obs, dim))
            
            for t_idx, t in enumerate(time_points):
                # Evaluate parameter functions at time t
                param_values = [pf(t) for pf in param_functions]
                
                # Simulate based on simulator type
                if sim_type == 'ecological':
                    # Population dynamics: logistic growth with noise
                    base = param_values[0] * np.exp(-param_values[1] * t)
                    values[t_idx, 0] = base + np.random.normal(0, 0.1)
                elif sim_type == 'epidemiological':
                    # SIR model approximation
                    s = param_values[0] * np.exp(-param_values[1] * t)
                    i = param_values[2] * (1 - np.exp(-param_values[3] * t))
                    values[t_idx, 0] = s + np.random.normal(0, 0.05)
                    values[t_idx, 1] = i + np.random.normal(0, 0.05)
                else:  # neuroscience
                    # Neuronal firing rate with oscillations
                    freq = param_values[0]
                    phase = param_values[1]
                    base_rate = param_values[2]
                    values[t_idx, 0] = base_rate + np.sin(2 * np.pi * freq * t + phase) + np.random.normal(0, 0.1)
                    values[t_idx, 1] = base_rate + np.cos(2 * np.pi * freq * t + phase) + np.random.normal(0, 0.1)
                    values[t_idx, 2] = base_rate + 0.5 * np.sin(4 * np.pi * freq * t + 2 * phase) + np.random.normal(0, 0.1)
            
            # Introduce missing observations (10-30% missing)
            mask = np.random.choice([True, False], size=(n_obs, dim), p=[0.7, 0.3])
            values[~mask] = np.nan  # Mark missing as NaN
            
            # Store data sample
            batch_data.append({
                'time': time_points,
                'values': values,
                'mask': mask
            })
            
            # Store metadata
            batch_metadata.append({
                'simulator_type': sim_type,
                'n_observations': n_obs,
                'dimensions': dim,
                'missing_rate': np.mean(~mask)
            })
        
        return {
            'parameters': batch_parameters,
            'data': batch_data,
            'metadata': batch_metadata
        }