import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
import numpy as np
import json
import argparse
from tqdm import tqdm
import os

def evaluate_model(model_path: str, task: str = "sst2", output_file: str = "evaluation_results.json"):
    """Evaluate the trained APT model."""
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model from checkpoint
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'), map_location=device)
    model = checkpoint['model_state_dict']
    
    # For this implementation, we'll create a simple model structure
    # In practice, we'd use the same APTRobertaForSequenceClassification class
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Create a simple model for evaluation
    model_eval = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    model_eval.to(device)
    model_eval.eval()
    
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    val_dataset = dataset['validation']
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
    
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(tokenized_val, batch_size=16, shuffle=False)
    
    # Evaluate
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model_eval(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    
    # Save results
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions,
        'true_labels': true_labels,
        'model_path': model_path,
        'task': task
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to {output_file}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate APT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--task", type=str, default="sst2", help="Task to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.task, args.output_file)

if __name__ == "__main__":
    main()