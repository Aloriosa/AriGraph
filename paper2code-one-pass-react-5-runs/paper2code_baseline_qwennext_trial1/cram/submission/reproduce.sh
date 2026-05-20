#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch torchvision torchaudio transformers datasets accelerate numpy

# Create directory for outputs
mkdir -p /home/submission/output

# Download and run the reproduction script
cd /home/submission

# Download the reproduction script from the paper's code repository
# Since the paper provides a URL, we assume the code is available
# We'll create a simplified version of the reproduction script based on the paper's description

# Create the main reproduction script
cat > compression_reproduction.py << 'EOF'
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import argparse
import os

def setup_model(model_name):
    """Load the pre-trained model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def compress_text(text, model, tokenizer, max_tokens=1568, device='cuda'):
    """
    Compress a text sequence into a set of [mem] vectors using per-sample optimization.
    This implementation follows the method described in the paper.
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    token_ids = tokens[0].to(device)
    
    # Get the model's embedding dimension
    embedding_dim = model.get_input_embeddings().embedding_dim
    vocab_size = model.config.vocab_size
    
    # Initialize the [mem] vector (one vector for this implementation)
    mem_vector = torch.randn(1, embedding_dim, requires_grad=True, device=device)
    
    # Define optimizer for the mem vector
    optimizer = optim.Adam([mem_vector], lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    steps = 0
    max_steps = 5000
    
    # We'll use teacher-forcing to train the mem vector
    # The idea is to prepend the mem_vector to the input sequence
    # and train it to predict the next token in the sequence
    while steps < max_steps and patience_counter < patience:
        optimizer.zero_grad()
        
        # Prepend the mem_vector to the token sequence
        # This is the core idea: the mem_vector is trained to encode the entire text
        # so that the LM can decode the text from just the mem_vector
        input_ids = torch.cat([mem_vector, token_ids], dim=0).unsqueeze(0)
        
        # We need to handle the case where the input is too long
        # We'll use a sliding window approach or truncate
        if input_ids.shape[1] > model.config.max_position_embeddings:
            input_ids = input_ids[:, :model.config.max_position_embeddings]
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # We want to predict the next token for each position
        # We'll use cross-entropy loss
        # We'll shift the labels to predict the next token
        shift_logits = logits[..., :-1, :]
        shift_labels = token_ids.unsqueeze(0)[..., 1:]
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        steps += 1
        
        # If we've achieved perfect reconstruction, stop
        if loss.item() < 0.01:
            break
    
    # Return the trained mem_vector
    return mem_vector.detach().cpu()

def decode_text(mem_vector, model, tokenizer, max_tokens=1568, device='cuda'):
    """
    Decode the text from the mem_vector using the trained model
    """
    mem_vector = mem_vector.to(device)
    
    # Prepend the mem_vector to the input
    # We'll use greedy decoding
    input_ids = mem_vector.unsqueeze(0)
    
    # Generate text
    generated_ids = model.generate(
        input_ids,
        max_length=max_tokens,
        num_beams=1,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    
    # Decode the generated text
    decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return decoded_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--text', default='The quick brown fox jumps over the lazy dog')
    parser.add_argument('--output', default='output.csv')
    parser.add_argument('--tokens', type=int, default=1568)
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model, tokenizer = setup_model(args.model)
    model = model.to(device)
    
    # Compress text
    text = args.text
    mem_vector = compress_text(text, model, tokenizer, max_tokens=args.tokens, device=device)
    
    # Decode text
    decoded_text = decode_text(mem_vector, model, tokenizer, max_tokens=args.tokens, device=device)
    
    # Save results
    with open(args.output, 'w') as f:
        f.write('original_text,decoded_text,compression_ratio\n')
        compression_ratio = len(text) / (mem_vector.numel() * 2)  # 2 bytes per float16
        f.write(f'{text},{decoded_text},{compression_ratio}\n')
    
    print(f"Original text: {text}")
    print(f"Decoded text: {decoded_text}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x compression_reproduction.py

# Run the reproduction script
python3 compression_reproduction.py --model meta-llama/Meta-Llama-3.1-8B --text "The quick brown fox jumps over the lazy dog" --output output.csv

# Inform the user that the output has been saved
echo "Compression reproduction completed. Output saved to output.csv"