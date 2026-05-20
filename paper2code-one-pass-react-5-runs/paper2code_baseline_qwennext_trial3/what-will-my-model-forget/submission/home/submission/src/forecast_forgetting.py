import json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import argparse
import os

class RepresentationBasedForecastingModel(nn.Module):
    """
    Representation-based forecasting model for forgetting prediction.
    This model uses inner products of trainable representations of examples to predict forgetting.
    Implements the model from Section 3.3 of the paper.
    """
    
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=1):
        super(RepresentationBasedForecastingModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learnable encoding function h that maps examples to low-dimensional representations
        # In practice, this would be a neural network (e.g., MLP)
        self.encoding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Learnable projection to scalar for inner product
        self.inner_product_projection = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, example1_repr, example2_repr):
        """
        Forward pass to predict probability of forgetting
        Implements: g(<x_i, y_i>, <x_j, y_j>) = σ(h(x_j, y_j) * h(x_i, y_i)^T)
        """
        # Encode examples to representations
        h_x1 = self.encoding_network(example1_repr)
        h_x2 = self.encoding_network(example2_repr)
        
        # Compute inner product of representations
        # This is the key insight: the inner product measures similarity
        # which correlates with forgetting
        inner_product = torch.sum(h_x1 * h_x2, dim=1, keepdim=True)
        
        # Apply sigmoid to get probability
        probability = self.sigmoid(inner_product)
        
        return probability

def load_data(filepath):
    """
    Load example data from JSON file
    Format: List of dicts with keys: 'input', 'output', 'is_forgotten'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def preprocess_examples(data, tokenizer, max_length=64):
    """
    Preprocess examples using tokenizer
    Returns: List of tensors of encoded representations
    """
    representations = []
    
    for example in data:
        # Combine input and output text for representation
        text = example['input'] + " " + example['output']
        
        # Tokenize
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Get representation from BERT model
        with torch.no_grad():
            outputs = bert_model(**encoded)
            # Use CLS token representation
            representation = outputs.last_hidden_state[:, 0, :]  # Shape: [1, 768]
            representations.append(representation)
    
    return representations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default='/home/submission/data/sample_data.json')
    parser.add_argument('--output', type=str, default='/home/submission/results/forecasting_results.json')
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.input_data)
    print(f"Loaded {len(data)} examples")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    global bert_model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Preprocess examples
    representations = preprocess_examples(data, tokenizer)
    
    # Initialize forecasting model
    model = RepresentationBasedForecastingModel(input_dim=768)
    
    # Training loop (for demonstration, we'll use the provided data)
    # In practice, this would be trained on a larger dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Training loop
    model.train()
    print("Training representation-based forecasting model...")
    for epoch in range(5):  # 5 epochs for demonstration
        total_loss = 0
        for i in range(len(representations)):
            for j in range(len(representations)):
                if i == j:
                    continue
                
                # Create a batch of pairs
                repr_i = representations[i]
                repr_j = representations[j]
                
                # Create target: 1 if example j is forgotten when learning i
                # In real data, we would have this from experiments
                target = torch.tensor([float(data[i]['is_forgotten'])])
                
                # Forward pass
                prediction = model(repr_i, repr_j)
                loss = torch.nn.functional.binary_cross_entropy(prediction, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(representations):.4f}")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    correct = 0
    results = []
    
    with torch.no_grad():
        for i in range(len(data)):
            example_i = data[i]
            example_i_repr = representations[i]
            
            example_results = []
            for j in range(len(data)):
                example_j = data[j]
                example_j_repr = representations[j]
                
                # Predict probability of forgetting
                prob = model(example_i_repr, example_j_repr)
                predicted = 1 if prob.item() > 0.5 else 0
                actual = data[j]['is_forgotten']
                
                example_results.append({
                    'example_i': i,
                    'example_j': j,
                    'input_i': example_i['input'],
                    'output_i': example_i['output'],
                    'input_j': example_j['input'],
                    'output_j': example_j['output'],
                    'actual': actual,
                    'predicted': predicted,
                    'probability': prob.item()
                })
            
            results.extend(example_results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    total = len(results)
    predicted_forgotten = sum(r['predicted'] for r in results)
    actual_forgotten = sum(r['actual'] for r in results)
    
    print(f"\nSummary:")
    print(f"Total predictions: {total}")
    print(f"Actual forgotten: {actual_forgotten}")
    print(f"Predicted forgotten: {predicted_forgotten}")
    print(f"Accuracy: {100 * (predicted_forgotten == actual_forgotten) if total > 0 else 0:.1f}%")
    
    # Print some examples
    print("\nSome predictions:")
    for r in results[:5]:
        print(f"  {r['example_i']} -> {r['example_j']:2d}: {r['probability']:.3f} -> {r['predicted']} (actual: {r['actual']})")

if __name__ == "__main__":
    main()