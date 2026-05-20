import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import time
from tqdm import tqdm

class MemCompressor(nn.Module):
    def __init__(self, model_name, num_mem_vectors=1):
        super(MemCompressor, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.num_mem_vectors = num_mem_mem_vectors
        self.mem_vectors = nn.Parameter(torch.randn(num_mem_vectors, self.model.config.hidden_size))

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        mem_expanded = self.mem_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        new_input_ids = torch.cat([mem_expanded, self.model.get_input_embeddings()(input_ids)], dim=1)
        outputs = self.model(inputs_embeds=new_input_ids, attention_mask=attention_mask)
        return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B', help='Model name')
    parser.add_argument('--input', type=str, default='data/pg19_sample.txt', help='Input file')
    parser.add_argument('--output', type=str, default='results/compression_results.json', help='Output file')
    parser.add_argument('--max_tokens', type=int, default=1568, help='Max tokens to compress')
    parser.add_argument('--num_vectors', type=int, default=1, help='Number of mem vectors')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # Load texts
    with open(args.input, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(texts)} texts")

    # Tokenize
    tokenized_texts = []
    for text in texts:
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        if len(tokens) > args.max_tokens:
            tokens = tokens[:args.max_tokens]
        tokenized_texts.append(tokens)

    # Initialize compressor
    compressor = MemCompressor(args.model, num_mem_vectors=args.num_vectors)
    compressor.to(device)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Training loop
    print("Starting training...")
    results = []
    for idx, tokens in enumerate(tqdm(tokenized_texts)):
        if len(tokens) == 0:
            continue
        # Convert to tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        input_embeds = model.get_input_embeddings()(input_tensor)
        batch_size = input_tensor.size(0)
        mem_expanded = compressor.mem_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        new_input_embeds = torch.cat([mem_expanded, input_embeds], dim=1)
        # Forward
        with torch.no_grad():
            outputs = model(inputs_embeds=new_input_embeds)
        logits = outputs.last_hidden_state
        # Compute loss
        shift_logits = logits[..., :-1, :]
        shift_labels = input_tensor[..., 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.contiguous().view(-1, logits.size(-1)), shift_labels.contiguous().view(-1))
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Check accuracy
        pred = torch.argmax(shift_logits, dim=-1)
        acc = (pred == shift_labels).float().mean()
        if acc.item() > 0.99:
            results.append({'text_id': idx, 'tokens': len(tokens), 'accuracy': acc.item()})
        if len(results) >= 10:
            break

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()