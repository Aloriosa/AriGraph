"""
Utilities for loading datasets.
"""

import datasets
from datasets import load_dataset
from typing import Dict, Any

from .config import DATASETS


def load_dataset_by_name(name: str, split: str):
    """Load a HuggingFace dataset given the name and split."""
    return load_dataset(name, split=split)


def get_dataset_info(dataset_key: str) -> Dict[str, Any]:
    """Return dataset configuration info."""
    return DATASETS[dataset_key]