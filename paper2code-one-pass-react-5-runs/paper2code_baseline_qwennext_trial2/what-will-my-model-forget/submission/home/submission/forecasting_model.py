import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
from data_generator import generate_dataset

class LogitChangeForecastingModel(nn.Module):
    """
    Partially interpretable forecasting model based on logit-change transfer
    as described in Section 3.2 of the paper.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=1):
        super(LogitChangeForecastingModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the neural network layers
        # This is a simplified version of the kernel approximation in Eqn. 2
        # where Θ(x_j, x_i)Θ^(-1)(x_i, x_i) is approximated by a learnable transformation
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        
    def forward(self, online_repr, upstream_repr):
        """
        Forward pass
        online_repr: [batch_size, input_dim] - representation of online learned example
        upstream_repr: [batch_size, input_dim] - representation of upstream pretraining example
        """
        # Concatenate the representations
        # This is a simplified version of the kernel approximation
        # where we're learning how much logit change is transferred based on similarity
        combined = torch.cat([online_repr, upstream_repr], dim=1)
        
        # Apply the neural network layers
        x = self.linear1(combined)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        
        return x

class RepresentationForecastingModel(nn.Module):
    """
    Black-box forecasting model based on inner products of representations
    as described in Section 3.3 of the paper.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=1):
        super(RepresentationForecastingModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the neural network layers
        # This implements g(<x_i, y_i>, <x_j, y_j>) = σ(h(x_j, y_j)h(x_i, y_i)^T)
        # where h is a trainable encoding function
        self.h = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        
    def forward(self, online_repr, upstream_repr):
        """
        Forward pass
        online_repr: [batch_size, input_dim] - representation of online learned example
        upstream_repr: [batch_size, input_dim] - representation of upstream pretraining example
        """
        # Compute the inner product of representations
        # This is a simplified version of the inner product h(x_j, y_j)h(x_i, y_i)^T
        # where h is a trainable encoding function
        online_proj = self.h(online_repr)
        upstream_proj = self.h(upstream_repr)
        
        # Compute the inner product
        # This is the similarity score
        inner_product = torch.sum(online_proj * upstream_proj, dim=1, keepdim=True)
        
        # Apply the final layer
        x = self.linear(inner_product)
        x = self.sigmoid(x)
        
        return x

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_type):
    """
    Train the forecasting model
    """
    best_val_f1 = 0
    train_losses = []
    val_f1s = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (online_repr, upstream_repr, target) in enumerate(train_loader):
            online_repr = online_repr.to(device)
            upstream_repr = upstream_repr.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(online_repr, upstream_repr)
            loss = nn.BCELoss()(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.round()
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += len(target)
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for online_repr, upstream_repr, target in val_loader:
                online_repr = online_repr.to(device)
                upstream_repr = upstream_repr.to(device)
                target = target.to(device)
                
                output = model(online_repr, upstream_repr)
            val_preds.extend(output.cpu().numpy().flatten())
            val_targets.extend(target.cpu().numpy().flatten())
        
        val_f1 = f1_score(val_targets, np.array(val_preds).round(), average='binary')
        val_f1s.append(val_f1)
        
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'models/best_{model_type}_model.pth")
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val F1: {val_f1:.4f}')
    
    return best_val_f1

def main():
    parser = argparse.ArgumentParser(description='Forecasting forgotten examples in language model refinement')
    parser.add_argument('--model_type', type=str, default='representation', choices=['logit_change', 'representation'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='models/model.pth')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate dataset
    print("Generating dataset...")
    train_data, val_data, test_data = generate_dataset()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    if args.model_type == 'logit_change':
        model = LogitChangeForecastingModel(input_dim=768, hidden_dim=512, output_dim=1)
    else:
        model = RepresentationForecastingModel(input_dim=768, hidden_dim=512, output_dim=1)
    
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train model
    print(f"Training {args.model_type} model...")
    best_val_f1 = train_model(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, args.model_type)
    
    # Save model
    print(f"Saving model to {args.output}")
    torch.save(model.state_dict(), args.output)
    
    print("Training complete!")
    
if __name__ == '__main__':
    main()