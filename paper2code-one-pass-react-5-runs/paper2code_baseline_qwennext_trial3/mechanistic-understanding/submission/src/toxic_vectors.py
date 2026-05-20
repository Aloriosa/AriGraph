import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

class ToxicityProbe(nn.Module):
    """Simple linear probe to identify toxicity vectors in language model representations."""
    
    def __init__(self, hidden_size):
        super(ToxicityProbe, self).__init__()
        self.probe = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        return self.probe(x)

class ToxicVectorExtractor:
    """Extract toxic vectors from language model representations."""
    
    def __init__(self, model_name='gpt2-medium'):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.probe = None
        self.toxic_vectors = []
        self.svd_vectors = []
        
    def load_data(self, data_path):
        """Load toxicity dataset."""
        import pandas as pd
        df = pd.read_csv(data_path)
        # Assume columns: 'comment', 'toxic'
        df = df.dropna()
        df = df.head(1000)  # Use subset for efficiency
        self.data = df
        
    def extract_residual_vectors(self, max_samples=500):
        """Extract residual vectors from model."""
        print("Extracting residual vectors...")
        
        # Use a sample of data
        samples = self.data.sample(n=min(max_samples, len(self.data)))
        residual_vectors = []
        
        for idx, row in samples.iterrows():
            # Tokenize text
            text = row['comment']
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get last layer residual stream
            residual_stream = outputs.last_hidden_state
            # Average over sequence length
            avg_stream = torch.mean(residual_stream, dim=1)
            residual_vectors.append(avg_stream.numpy())
        
        return np.vstack(residual_vectors)
    
    def train_probe(self, data_path, save_path='probe_model.pkl'):
        """Train toxicity probe on residual stream."""
        print("Training toxicity probe...")
        self.load_data(data_path)
        residual_vectors = self.extract_residual_vectors()
        
        # Create labels
        labels = self.data['toxic'].astype(int).values
        labels = labels[:len(residual_vectors)]
        
        # Train logistic regression probe
        self.probe = LogisticRegression(max_iter=1000)
        self.probe.fit(residual_vectors, labels)
        
        # Save probe
        with open(save_path, 'wb') as f:
            pickle.dump(self.probe, f)
        
        print(f"Probe trained and saved to {save_path}")
    
    def extract_toxic_vectors(self, num_vectors=128):
        """Extract toxic vectors based on probe similarity."""
        print("Extracting toxic vectors...")
        
        # Get MLP value vectors from model
        # In actual implementation, we would extract MLP value vectors
        # For reproduction, we'll simulate this
        hidden_size = self.model.config.hidden_size
        
        # Get all value vectors from MLP blocks
        # This is a simplified version - in practice we'd extract from specific layers
        value_vectors = []
        
        # Simulate extracting value vectors from MLP blocks
        # In actual model, we would extract from MLP blocks
        for name, param in self.model.named_parameters():
            if 'mlp' in name and 'weight' in name and 'dense' in name:
                # Extract value vectors
                if len(param.shape) == 2 and param.shape[0] == param.shape[1]:
                    # This is a value vector
                    value_vectors.append(param.detach().numpy())
        
        # Use probe to find toxic vectors
        probe_weights = self.probe.coef_[0]
        
        # Find vectors with high similarity to probe weights
        toxic_vectors = []
        for vec in value_vectors:
            # Normalize
            if len(vec.shape) == 1:
                vec = vec.reshape(-1)
            if len(vec) != len(probe_weights):
                continue
            similarity = np.dot(vec, probe_weights) / (np.linalg.norm(vec) * np.linalg.norm(probe_weights))
            if similarity > 0.1:  # High similarity
                toxic_vectors.append(vec)
        
        # Select top toxic vectors
        toxic_vectors = toxic_vectors[:num_vectors]
        self.toxic_vectors = np.array(toxic_vectors)
        
        # Apply SVD on toxic vectors
        if len(self.toxic_vectors) > 0:
            u, s, vh = np.linalg.svd(self.toxic_vectors)
            self.svd_vectors = u
        
        print(f"Extracted {len(self.toxic_vectors)} toxic vectors")
        return self.toxic_vectors, self.svd_vectors

# Example usage
if __name__ == "__main__":
    extractor = ToxicVectorExtractor()
    # extractor.train_probe('jigsaw_toxic_comments.csv')
    # vectors, svd = extractor.extract_toxic_vectors()
    print("Toxic vector extraction module loaded")