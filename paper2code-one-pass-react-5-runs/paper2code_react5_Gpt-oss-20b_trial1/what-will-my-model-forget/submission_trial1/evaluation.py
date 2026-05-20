"""Utility functions for evaluating forecasting models."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def evaluate_forecasting(
    predictions: dict,  # {j: 0/1}
    ground_truth: dict,  # {j: 0/1}
):
    """Return F1, precision, recall for a single forecasting method."""
    y_true = np.array([ground_truth[j] for j in sorted(ground_truth)])
    y_pred = np.array([predictions[j] for j in sorted(predictions)])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"F1": f1 * 100.0, "Precision": precision * 100.0, "Recall": recall * 100.0}