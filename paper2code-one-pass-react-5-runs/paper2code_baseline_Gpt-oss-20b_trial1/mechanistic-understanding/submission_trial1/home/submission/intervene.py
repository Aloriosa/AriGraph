"""
Hook to subtract a scaled probe vector from the residual stream
(e.g., the final hidden state before the language‑model output).
"""

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def intervene_and_generate(model, tokenizer, prompt, alpha=0.3,
                           device="cpu", max_new_tokens=20):
    """
    Generates text while subtracting `alpha * probe_vector` from the
    last hidden state of the model. The probe vector is expected to be
    a model‑level global variable `PROBE_WEIGHTS` (torch.Tensor).
    """
    # Ensure the probe vector is on the correct device
    global PROBE_WEIGHTS
    probe = PROBE_WEIGHTS.to(device)

    # Define a forward hook that modifies the hidden state
    def hook(module, input, output):
        # `output` is the hidden states of shape (batch, seq_len, hidden_dim)
        # We subtract the probe from the last time step (the current state)
        # before the next token is produced.
        # For GPT‑2, the hidden state is used to compute logits via the
        # unembedding layer. We modify the *last* hidden state in `output`.
        # Since generate runs in an autoregressive loop, we hook on the
        # `transformer` module (the whole transformer stack).
        # Here we simply modify the last hidden state of the whole output.
        # Note: `output` has shape (batch, seq_len, hidden_dim)
        output[:, -1, :] = output[:, -1, :] - alpha * probe
        return output

    # Register hook on the last transformer layer (the one that feeds into logits)
    handle = model.transformer.h[-1].register_forward_hook(hook)

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    handle.remove()  # clean up

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)