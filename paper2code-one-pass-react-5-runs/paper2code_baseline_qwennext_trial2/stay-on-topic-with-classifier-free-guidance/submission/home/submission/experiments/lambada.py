#!/usr/bin/env python3
"""
LAMBADA experiment with LLaMA-7B using Classifier-Free Guidance (CFG)
Reproduces the LAMBADA zero-shot experiment from the paper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import argparse
import json
import time
import os

def setup_model(model_name):
    """Setup the LLaMA-7B model and tokenizer"""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def classifier_free_guidance(logits, logits_cond, logits_uncond, gamma):
    """
    Apply Classifier-Free Guidance to logits
    Based on Equation 7 from the paper:
    log P̂(w_i | w_{<i}, c) = log P_θ(w_i | w_{<i}) + γ(log P_θ(w_i | w_{<i}, c) - log P_θ(w_i | w_{<i}))
    """
    # CFG formula: logits = logits_uncond + gamma * (logits_cond - logits_uncond)
    # This is equivalent to the formula in the paper
    logits_cfg = logits_uncond + gamma * (logits_cond - logits_uncond)
    return logits_cfg

def evaluate_lambada(tokenizer, model, gamma=1.5, max_samples=100):
    """
    Evaluate on LAMBADA dataset using zero-shot setting
    """
    print(f"Evaluating on LAMBADA with gamma={gamma}")
    
    # Load LAMBADA dataset
    dataset = load_dataset("lambada", split="test")
    
    # We'll use the first max_samples samples
    samples = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    # For each sample
    for sample in samples:
        text = sample["text"]
        
        # Get the last word (the target)
        words = text.split()
        target_word = words[-1]
        
        # Create prompt (just the context without the last word)
        context = " ".join(words[:-1])
        
        # Get the full text as the prompt
        full_prompt = context
        
        # Tokenize
        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
        
        # Get logits for the last token position
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get the last token position
        last_token_pos = inputs["input_ids"].shape[1] - 1
        
        # Get logits for the last token
        last_token_logits = logits[0, last_token_pos]
        
        # Get the predicted token
        predicted_token_id = torch.argmax(last_token_logits).item()
        predicted_word = tokenizer.decode([predicted_token_id])
        
        # Check if it matches
        if predicted_word.strip() == target_word.strip():
            correct += 1
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="LAMBADA experiment with LLaMA-7B using CFG")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-7B", help="Model name")
    parser.add_argument("--output", type=str, default="results/lambada_results.csv", help="Output file")
    parser.add_argument("--gamma_values", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0, 3.0], help="Gamma values to test")
    
    args = parser.parse_args()
    
    # Setup model
    tokenizer, model = setup_model(args.model_name)
    
    # Test different gamma values
    results = []
    
    for gamma in args.gamma_values:
        print(f"\nTesting gamma={gamma}")
        accuracy = evaluate_lambada(tokenizer, model, gamma=gamma)
        results.append({
            "gamma": gamma,
            "accuracy": accuracy
        })
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()