#!/usr/bin/env python
"""
Utility helpers used in run_experiment.py.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_pooled_representation(model, tokenizer, text, max_len=128):
    inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    with torch.no_grad():
        encoder = model.module.encoder if hasattr(model, "module") else model.encoder
        encoder_outputs = encoder(input_ids)
        pooled = encoder_outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
    return pooled.squeeze(0)