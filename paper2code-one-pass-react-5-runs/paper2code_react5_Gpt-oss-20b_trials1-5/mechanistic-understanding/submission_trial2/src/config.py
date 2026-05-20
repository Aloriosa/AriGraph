# Configuration parameters used throughout the repo.
# These are intentionally simple so the script can run on a single GPU.

MODEL_NAMES = {
    "gpt2": "gpt2-medium",
    "llama2": "meta-llama/Llama-2-7b-hf",
}
# Maximum sequence length used for training/evaluation
MAX_SEQ_LENGTH = 128

# Training hyper‑parameters
EPOCHS = 1          # keep tiny – this is a demonstration, not a full experiment
BATCH_SIZE = 4      # batch size per GPU (adjust if out‑of‑memory)
LEARNING_RATE = 1e-6
BETAS = (0.9, 0.999)   # AdamW betas
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# DPO hyper‑parameters
DPO_BETA = 0.1          # temperature in the DPO loss
DPO_BATCH_SIZE = 4

# Tokenizer / generation parameters
NUM_CONT = 5            # number of continuations to sample when searching for a toxic one
CONT_LENGTH = 20        # length of each continuation
TOP_K = 10
TOP_P = 0.95

# Toxicity classifier
TOXICITY_MODEL = "unitary/toxic-bert"

# Evaluation
TEST_SPLIT = "test"