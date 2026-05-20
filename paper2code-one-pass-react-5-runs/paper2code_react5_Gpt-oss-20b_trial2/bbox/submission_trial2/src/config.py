"""
Global hyper‑parameters.
"""

# Adapter model
ADAPTER_HIDDEN_DIM = 256
ADAPTER_OUTPUT_DIM = 1  # scalar energy

# Training
TRAIN_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 5e-6
MAX_SEQ_LEN = 512
TRAIN_SIZE = 200  # number of training examples

# Adapter size (scaled relative to hidden dim)
# 0.1 means 10% of hidden dimension
ADAPTER_SIZE = 0.1

# LLM for generation
LLM_NAME = "mistralai/Mixtral-8x7B-v0.1"
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 1.0

# Online adaptation
BEAM_SIZE = 3
NUM_CANDIDATES = 5