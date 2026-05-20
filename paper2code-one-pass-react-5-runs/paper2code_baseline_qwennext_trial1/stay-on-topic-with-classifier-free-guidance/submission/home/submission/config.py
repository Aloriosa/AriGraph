"""
Configuration for experiments
"""

# Model configurations
class Config:
    # Model names
    MODEL_NAMES = {
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
        'llama-7b': 'huggyllama/llama-7b'
    }
    
    # CFG parameters
    CFG_GAMMAS = [1.0, 1.25, 1.5, 1.75, 2.0]
    
    # Evaluation parameters
    MAX_LENGTH = 1024
    BATCH_SIZE = 4
    
    # Results directory
    RESULTS_DIR = 'results'
    
    # Data files
    LAMBADA_TEST = 'data/lambada_test.jsonl'
    HUMAN_EVAL = 'data/humaneval.json'
    
    # Model paths
    MODEL_PATHS = {
        'gpt2-medium': 'models/gpt2-medium/pytorch_model.bin',
        'gpt2-large': 'models/gpt2-large/pytorch_model.bin',
        'llama-7b': 'models/llama-7b/pytorch_model.bin'
    }