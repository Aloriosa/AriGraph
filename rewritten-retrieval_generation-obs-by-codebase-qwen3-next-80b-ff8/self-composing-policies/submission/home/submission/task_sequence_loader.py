#!/usr/bin/env python3
"""
Task sequence loader for Meta-World benchmarks.
Implements the 20-task sequence with 10 unique tasks and 10 duplicates as described in the paper.
"""
import metaworld
import numpy as np
from typing import List
import random

def get_task_sequence(sequence_name: str) -> List:
    """
    Get the task sequence for the specified benchmark.
    
    Args:
        sequence_name: Name of the task sequence to load
        
    Returns:
        List of MetaWorld tasks
    """
    if sequence_name == "meta_world_20":
        # Define the 20-task sequence with 10 unique tasks and 10 duplicates
        # This follows the CW20 benchmark from the paper
        unique_tasks = [
            "hammer-v2", "push-wall-v2", "faucet-close-v2", 
            "push-back-v2", "stick-pull-v2", "handle-press-side-v2", 
            "push-v2", "shelf-place-v2", "window-close-v2", 
            "peg-unplug-side-v2"
        ]
        
        # Create the sequence: 10 unique tasks followed by 10 duplicates in random order
        task_sequence = []
        
        # Add unique tasks
        for task_name in unique_tasks:
            task = metaworld.MT1(task_name).train_classes[task_name]()
            task_sequence.append(task)
        
        # Add duplicates (randomly shuffled)
        duplicate_tasks = unique_tasks.copy()
        random.shuffle(duplicate_tasks)
        for task_name in duplicate_tasks:
            task = metaworld.MT1(task_name).train_classes[task_name]()
            task_sequence.append(task)
        
        return task_sequence
    
    elif sequence_name == "meta_world_10":
        # 10 unique tasks only
        unique_tasks = [
            "hammer-v2", "push-wall-v2", "faucet-close-v2", 
            "push-back-v2", "stick-pull-v2", "handle-press-side-v2", 
            "push-v2", "shelf-place-v2", "window-close-v2", 
            "peg-unplug-side-v2"
        ]
        
        task_sequence = []
        for task_name in unique_tasks:
            task = metaworld.MT1(task_name).train_classes[task_name]()
            task_sequence.append(task)
        
        return task_sequence
    
    elif sequence_name == "atari_spaceinvaders":
        # For Atari, we'd need a different implementation
        # This is a placeholder
        return []
    
    else:
        raise ValueError(f"Unknown task sequence: {sequence_name}")