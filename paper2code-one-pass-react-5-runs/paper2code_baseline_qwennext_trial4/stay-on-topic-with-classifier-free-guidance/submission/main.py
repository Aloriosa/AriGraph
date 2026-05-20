import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import csv
import os
import argparse

def classifier_free_guidance(logits_conditional, logits_unconditional, guidance_weight=1.5):
    """
    Apply Classifier-Free Guidance to logits.
    Formula from paper: 
    log P_modified = log P_conditional + γ * (log P_conditional - log P_unconditional)
    
    Args:
        logits_conditional: logits from model with prompt
        logits_unconditional: logits from model without prompt
        guidance_weight: gamma value
    Returns:
        modified logits
    """
    # Apply CFG formula
    modified_logits = logits_conditional + guidance_weight * (logits_conditional - logits_unconditional)
    return modified_logits

def generate_with_cfg(model, tokenizer, prompt, max_length=50, guidance_weight=1.5):
    """
    Generate text using Classifier-Free Guidance.
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        prompt: input prompt
        max_length: max generation length
        guidance_weight: gamma value
    Returns:
        generated text
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to('cuda') if torch.cuda.is_available() else input_ids
    
    # Generate unconditional output (without prompt)
    # We use an empty string as the unconditional prompt
    empty_input_ids = tokenizer.encode("", return_tensors='pt')
    empty_input_ids = empty_input_ids.to('cuda') if torch.cuda.is_available() else empty_input_ids
    
    # Generate conditional output
    with torch.no_grad():
        # Get logits for conditional generation
        conditional_outputs = model(input_ids)
        conditional_logits = conditional_outputs.logits[0, -1, :]  # Last token logits
        conditional_probs = F.softmax(conditional_logits, dim=-1)
        
        # Get logits for unconditional generation
        unconditional_outputs = model(empty_input_ids)
        unconditional_logits = unconditional_outputs.logits[0, -1, :]  # Last token logits
        unconditional_probs = F.softmax(unconditional_logits, dim=-1)
        
        # Apply CFG
        modified_logits = classifier_free_guidance(conditional_logits, unconditional_logits, guidance_weight)
        
        # Sample from modified logits
        modified_probs = F.softmax(modified_logits, dim=-1)
        next_token = torch.multinomial(modified_probs, num_samples=1)
    
    # Generate full sequence
    generated_ids = input_ids.clone()
    for _ in range(max_length - 1):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[0, -1, :]
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Reproduce Classifier-Free Guidance results')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--word', type=str, default='strawberry', help='Word to count r\'s in')
    parser.add_argument('--output', type=str, default='results/output.csv', help='Output file')
    parser.add_argument('--gamma', type=float, default=1.5, help='Guidance weight')
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    # Count 'r's in word
    r_count = args.word.lower().count('r')
    print(f"'{args.word}' has {r_count} 'r'(s).")
    
    # Generate with CFG
    prompt = f"Count the number of 'r's in the word '{args.word}'."
    generated_text = generate_with_cfg(model, tokenizer, prompt, max_length=10, guidance_weight=args.gamma)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["word", "r count", "gamma", "generated_text"])
        writer.writerow([args.word, r_count, args.gamma, generated_text])
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()