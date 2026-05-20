import torch
import torch.nn as nn
import torch.nn.functional as F

class ThresholdForecaster:
    """Predict forgetting if an example is forgotten in >= gamma fraction of training errors."""
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        self.forgot_counts = None
        self.total = None

    def fit(self, pt_examples, forgetting_matrix):
        """
        pt_examples: list of pretrain example ids (int)
        forgetting_matrix: 2D numpy array shape (len(D_R_train), len(pt_examples))
                           1 if pt_example is forgotten by that online example
        """
        self.forgot_counts = forgetting_matrix.sum(axis=0)
        self.total = forgetting_matrix.shape[0]
        self.gamma = self.gamma  # keep as is

    def predict(self, pt_example_ids):
        scores = self.forgot_counts[pt_example_ids] / self.total
        return (scores >= self.gamma).astype(int)

class LogitForecaster(nn.Module):
    """
    Trainable logit‑based forecasting model.
    Encodes examples with a small MLP and uses inner product as kernel.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def encode(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden]
        # simple mean pooling
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        return self.encoder(pooled)  # [batch, 64]

    def predict_logit(self, i_repr, j_repr, i_logits, j_logits_before):
        """
        i_repr, j_repr: [batch, 64]
        i_logits: logits of online example after update [batch, vocab]
        j_logits_before: logits of pretrain example before update [batch, vocab]
        Returns predicted logits for j after update.
        """
        # kernel = i_repr @ j_repr.T  -> scalar per pair
        k = torch.sum(i_repr * j_repr, dim=1, keepdim=True)  # [batch,1]
        delta_i = i_logits - i_logits.detach()  # we use detached as placeholder
        # Simplified: predicted change = k * delta_i
        pred_change = k * delta_i  # broadcast
        return j_logits_before + pred_change

    def forward(self, i_repr, j_repr, i_logits, j_logits_before):
        # For training we compute loss on the correct token
        pred_j_logits = self.predict_logit(i_repr, j_repr, i_logits, j_logits_before)
        return pred_j_logits

class RepresentationForecaster(nn.Module):
    """
    Black‑box representation‑based forecasting: logistic regression on inner product + bias.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.linear = nn.Linear(64, 1)

    def encode(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        return self.encoder(pooled)  # [batch, 64]

    def forward(self, i_repr, j_repr, bias):
        # dot product + bias -> logits
        dot = torch.sum(i_repr * j_repr, dim=1, keepdim=True)  # [batch,1]
        logits = dot + bias  # bias broadcast
        return logits.squeeze(1)  # [batch]