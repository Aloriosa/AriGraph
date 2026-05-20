import logging
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


def get_example_repr(
    model: AutoModelForSeq2SeqLM,
    tokenizer,
    example: dict,
    max_length: int = 128,
) -> np.ndarray:
    """
    Return a fixed‑size representation for an example.
    We use the mean of the encoder last hidden states.
    """
    model.eval()
    inp = f"question: {example['question']} context: {example['context']}"
    input_ids = tokenizer(inp, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    input_ids = input_ids.input_ids.to(model.device)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids)
    hidden = encoder_outputs.last_hidden_state  # shape [1, seq_len, hidden]
    # mean over seq_len dimension
    repr_vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return repr_vec


def build_feature_vectors(
    error_repr: np.ndarray,
    pt_repr: np.ndarray,
) -> np.ndarray:
    """
    For a pair (error, pt example) return a scalar feature: dot product.
    """
    return np.array([float(np.dot(error_repr, pt_repr))])


def train_logistic_regression(
    error_reprs: List[np.ndarray],
    pt_reprs: List[np.ndarray],
    labels: List[int],
) -> LogisticRegression:
    """
    Train a binary classifier that predicts forgetting from the dot product
    of error and pre‑training representations.
    """
    X = []
    y = []
    for er, pr, lbl in zip(error_reprs, pt_reprs, labels):
        X.append(build_feature_vectors(er, pr)[0])
        y.append(lbl)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    clf = LogisticRegression(solver="liblinear")
    clf.fit(X, y)
    return clf


def predict_forgetting(
    clf: LogisticRegression,
    error_repr: np.ndarray,
    pt_reprs: List[np.ndarray],
    threshold: float = 0.5,
) -> List[int]:
    """
    Return indices of pre‑training examples predicted to be forgotten.
    """
    feats = np.array([build_feature_vectors(error_repr, pr)[0] for pr in pt_reprs]).reshape(-1, 1)
    probs = clf.predict_proba(feats)[:, 1]
    return [i for i, p in enumerate(probs) if p >= threshold]