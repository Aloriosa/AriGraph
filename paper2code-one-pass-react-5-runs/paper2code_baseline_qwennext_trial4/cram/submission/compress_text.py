import os
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCompressor:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the TextCompressor with a pre-trained language model.
        
        Args:
            model_name (str): Name of the pre-trained model from Hugging Face.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        logger.info(f"Loaded model {model_name} on device {device}")
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer pad token set to {self.tokenizer.pad_token}")
        
    def encode_text(self, text: str, num_mem_vectors: int = 1, max_steps: int = 5000, lr: float = 0.01, weight_decay: float = 0.01, early_stop_threshold: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Compress a single text into a set of memory vectors.
        
        Args:
            text (str): Input text to compress.
            num_mem_vectors (int): Number of memory vectors to use.
            max_steps (int): Maximum optimization steps.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for AdamW optimizer.
            early_stop_threshold (float): Early stopping threshold for accuracy.
        
        Returns:
            Tuple[torch.Tensor, float]: Compressed memory vectors and final loss.
        """
        # Tokenize input text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        token_ids = tokens.squeeze(0)
        logger.info(f"Tokenized text with {len(token_ids)} tokens")
        
        # Initialize memory vectors randomly
        mem_vectors = torch.randn(num_mem_vectors, self.model.config.hidden_size, requires_grad=True, device=self.device)
        mem_vectors = torch.nn.Parameter(mem_vectors)
        optimizer = torch.optim.AdamW([mem_vectors], lr=lr, weight_decay=weight_decay)
        logger.info(f"Initialized {num_mem_vectors} memory vectors with shape {mem_vectors.shape}")
        
        best_loss = float('inf')
        early_stop_count = 0
        
        # Optimization loop
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Create full input: [mem_vectors] + [token_ids]
            input_ids = torch.cat([mem_vectors, token_ids.unsqueeze(0)], dim=1)
            
            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels).float().sum()
            accuracy = correct / (len(shift_labels))
            
            if step % 100 == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= 10:
                # Early stopping if loss doesn't improve for 10 steps
                    logger.info(f"Early stopping at step {step}")
                    break
        
        # Final evaluation for accuracy
        with torch.no_grad():
            input_ids = torch.cat([mem_vectors, token_ids.unsqueeze(0)], dim=1)
            outputs = self.model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels).float().sum()
            final_accuracy = correct / (len(shift_labels))
            
            logger.info(f"Final Accuracy: {final_accuracy.item():.4f}")
            
            # Return memory vectors and final loss
            return mem_vectors.detach().cpu(), loss.item()
    
    def decode_text(self, mem_vectors: torch.Tensor, max_length: int = 2048) -> str:
        """
        Decode text from memory vectors using the model.
        
        Args:
            mem_vectors (torch.Tensor): Memory vectors to decode from.
            max_length (int): Maximum length of generated text.
        
        Returns:
            str: Decoded text.
        """
        # Generate text from memory vectors
        mem_vectors = mem_vectors.to(self.device)
        input_ids = mem_vectors.unsqueeze(0)
        
        # Generate text
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=1,
            early_stopping=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Decode generated tokens
        decoded = self.tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
        return decoded

def main():
    parser = argparse.ArgumentParser(description="Reproduce text compression results from paper")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name")
    parser.add_argument("--text_source", type=str, default="pg19", choices=["pg19", "fanfics", "random"], help="Text source")
    parser.add_argument("--num_mem_vectors", type=int, default=1, help="Number of memory vectors")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--max_tokens", type=int, default=1568, help="Maximum tokens to compress")
    parser.add_argument("--num_texts", type=int, default=1, help="Number of texts to compress")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize compressor
    compressor = TextCompressor(args.model_name)
    
    # Load text data
    texts = []
    if args.text_source == "pg19":
        dataset = load_dataset("pg19", split="train")
        for item in dataset:
            if len(item["text"]) > args.max_tokens * 2:
                texts.append(item["text"][:args.max_tokens * 2])
            if len(texts) >= args.num_texts:
                break
    elif args.text_source == "fanfics":
        # Load fanfic texts (dummy implementation)
        # In real implementation, load from local files
        texts = ["This is a sample fanfic text." for _ in range(args.num_texts)]
    else:  # random
        # Generate random text
        vocab = "abcdefghijklmnopqrstuvwxyz "
        for _ in range(args.num_texts):
            text = "".join(np.random.choice(list(vocab), size=args.max_tokens))
            texts.append(text)
    
    # Compress texts
    results = []
    for i, text in enumerate(texts):
        logger.info(f"Compressing text {i + 1}/{len(texts)}")
        mem_vectors, loss = compressor.encode_text(text, num_mem_vectors=args.num_mem_vectors)
        
        # Decode text
        decoded_text = compressor.decode_text(mem_vectors, max_length=args.max_tokens)
        
        # Calculate metrics
        original_tokens = len(compressor.tokenizer.encode(text))
        decoded_tokens = len(compressor.tokenizer.encode(decoded_text))
        accuracy = 1.0 if text == decoded_text else 0.0
        
        results.append({
            "original_text": text,
            "decoded_text": decoded_text,
            "original_tokens": original_tokens,
            "decoded_tokens": decoded_tokens,
            "accuracy": accuracy,
            "compression_ratio": original_tokens / args.num_mem_vectors,
            "mem_vectors": mem_vectors.tolist()
        })
        
        # Save results
        output_file = os.path.join(args.output_dir, f"result_{i}.json")
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved result to {output_file}")
    
    # Save summary
    summary = {
        "model": args.model_name,
        "text_source": args.text_source,
        "num_texts": len(texts),
        "num_mem_vectors": args.num_mem_vectors,
        "max_tokens": args.max_tokens,
        "results": results
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Print final result
    final_accuracy = np.mean([r["accuracy"] for r in results])
    final_compression = np.mean([r["compression_ratio"] for r in results])
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Compression Ratio: {final_compression:.2f}x")
    
    # Save output.csv as required
    import csv
    output_csv = os.path.join(args.output_dir, "output.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "r count"])
        # We are compressing a text, but paper counts 'r's in 'strawberry'
        # For reproduction, we'll count 'r's in first text
        sample_text = texts[0] if texts else "strawberry"
        r_count = sample_text.lower().count('r')
        writer.writerow([sample_text, r_count])
    
    print(f"\n'output.csv' created with 'r count' for sample text: {sample_text} -> {r_count} 'r'(s).")

if __name__ == "__main__":
    main()