import pickle
import numpy as np
from typing import List, Tuple

Transition = Tuple[np.ndarray, int, np.ndarray, float]

def load_offline_dataset(path: str) -> List[Transition]:
    """Load the synthetic offline dataset from a pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_batches(dataset: List[Transition], batch_size: int):
    """Yield random batches from the offline dataset."""
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    for start in range(0, len(dataset), batch_size):
        batch_indices = indices[start:start+batch_size]
        batch = [dataset[i] for i in batch_indices]
        obs = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        next_obs = np.array([b[2] for b in batch], dtype=np.float32)
        rewards = np.array([b[3] for b in batch], dtype=np.float32)
        yield obs, actions, next_obs, rewards