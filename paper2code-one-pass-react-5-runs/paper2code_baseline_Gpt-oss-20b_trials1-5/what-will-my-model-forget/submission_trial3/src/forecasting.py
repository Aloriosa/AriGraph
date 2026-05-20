# src/forecasting.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

class ThresholdForecaster:
    """
    Predicts forgetting if the example has been forgotten more than
    a learned threshold gamma across the training set.
    """
    def __init__(self, forgetting_counts, threshold=None):
        """
        forgetting_counts: dict {pt_idx: count_of_forgetting}
        """
        self.forgetting_counts = forgetting_counts
        if threshold is None:
            self.threshold = self._find_optimal_threshold()
        else:
            self.threshold = threshold

    def _find_optimal_threshold(self):
        # brute force over possible thresholds
        counts = list(self.forgetting_counts.values())
        best_f1 = -1
        best_thr = 0
        for thr in range(max(counts) + 1):
            preds = [1 if c >= thr else 0 for c in counts]
            f1 = f1_score(self.y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        return best_thr

    def fit(self, y_true):
        """
        y_true: list of ground‑truth forgetting labels for PT examples
        """
        self.y_true = y_true

    def predict(self, pt_idx):
        return int(self.forgetting_counts.get(pt_idx, 0) >= self.threshold)


class LogitForecaster(nn.Module):
    """
    Trainable logit‑based forecasting model.
    Encodes examples with a small MLP, builds a kernel, and learns to
    predict the logit change transfer.
    """
    def __init__(self, hidden_dim=128, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def encode(self, hidden_states):
        """
        hidden_states: [batch, hidden_dim]
        Returns: [batch, proj_dim]
        """
        return self.encoder(hidden_states)

    def forward(self, h_i, h_j, delta_logit_i, label_j, target_label_j):
        """
        h_i: [proj_dim] representation of online example
        h_j: [proj_dim] representation of PT example
        delta_logit_i: [logit_dim] change in logits of online example
        label_j: current correct label id for PT example
        target_label_j: ground‑truth label id for PT example
        """
        # kernel: scalar
        kernel = torch.dot(h_j, h_i)  # [1]
        # predict change for PT example
        pred_delta = kernel.unsqueeze(-1) * delta_logit_i  # [logit_dim]
        # predicted logits for PT example
        # we use only the logits for the target label
        # for simplicity, we predict the logit of the correct token
        # (since in SST-2 we have 2 classes)
        # compute predicted logit difference for the correct class
        pred_logit_diff = pred_delta[:, label_j]
        # margin loss
        margin = 1.0
        # if target label is forgotten, we want the predicted logit of
        # correct class to be lower than the other class by margin
        # else we want it higher
        other_logit = pred_delta[:, 1 - label_j]
        loss = F.relu(margin + (-1)**(int(label_j == target_label_j)) *
                      (other_logit - pred_logit_diff)).mean()
        return loss

    def predict(self, h_i, h_j, delta_logit_i, label_j):
        kernel = torch.dot(h_j, h_i)
        pred_delta = kernel.unsqueeze(-1) * delta_logit_i
        pred_logit = pred_delta[:, label_j]
        # we decide forgetting if predicted logit is lower than the other
        other_logit = pred_delta[:, 1 - label_j]
        return int(pred_logit < other_logit)


class RepresentationForecaster(nn.Module):
    """
    Black‑box representation‑based forecasting model.
    Uses inner product of representations plus a bias term (frequency prior).
    """
    def __init__(self, hidden_dim=128, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)  # scalar output
        )
        self.b_prior = None  # will be set during training

    def encode(self, hidden_states):
        return self.encoder(hidden_states).squeeze(-1)  # [batch]

    def forward(self, h_i, h_j, b_j):
        """
        h_i, h_j: [proj_dim]
        b_j: bias term for PT example (scalar)
        """
        score = torch.dot(h_i, h_j) + b_j
        prob = torch.sigmoid(score)
        return prob

    def fit_prior(self, forgetting_counts, num_pt):
        """
        forgetting_counts: dict {pt_idx: count_of_forgetting}
        """
        freqs = torch.tensor(
            [forgetting_counts.get(i, 0) / num_pt for i in range(num_pt)],
            dtype=torch.float
        )
        # log odds
        eps = 1e-6
        self.b_prior = torch.log(freqs + eps) - torch.log(1 - freqs + eps)

    def predict(self, h_i, h_j, b_j):
        score = torch.dot(h_i, h_j) + b_j
        return int(torch.sigmoid(score) > 0.5)