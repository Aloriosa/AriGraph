"""Forecasting algorithms."""
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def encode_example(
    model,
    tokenizer,
    example: Dict,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Encode an example using the encoder part of T5.
    Returns a fixed‑size vector (mean pooling over hidden states).
    """
    input_text = f"Classify this news: {example['text']}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(DEVICE)

    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
        hidden_states = encoder_outputs.last_hidden_state  # (1, seq_len, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        length = torch.sum(mask, dim=1)
        mean_pooled = summed / (length + 1e-12)
    return mean_pooled.squeeze(0)  # (hidden,)


# --------------------------------------------------------------------------- #
# Threshold baseline
# --------------------------------------------------------------------------- #
def compute_forgetting_frequencies(
    D_PT: List[Dict],
    forgetting_labels: Dict[Tuple[int, int], int],  # (i, j) -> 0/1
    D_R_train: List[Dict],
) -> Dict[int, float]:
    """
    Compute forgetting frequency for each D_PT example across all training
    online updates. The key is the index of the D_PT example.
    """
    freq = {idx: 0 for idx in range(len(D_PT))}
    for (i, j), z in forgetting_labels.items():
        if z == 1:
            freq[j] += 1
    # Normalize by number of training online examples
    total = len(D_R_train)
    freq_norm = {j: f / total for j, f in freq.items()}
    return freq_norm


def threshold_baseline(
    D_PT: List[Dict],
    D_R_train: List[Dict],
    forgetting_labels: Dict[Tuple[int, int], int],
    gamma: float = 0.05,
) -> Dict[int, int]:
    """
    Predict that a D_PT example j is forgotten if its forgetting frequency
    exceeds gamma.
    """
    freq = compute_forgetting_frequencies(D_PT, forgetting_labels, D_R_train)
    preds = {j: 1 if f >= gamma else 0 for j, f in freq.items()}
    return preds


# --------------------------------------------------------------------------- #
# Representation‑based predictor
# --------------------------------------------------------------------------- #
class RepresentationForecaster:
    """
    Train a logistic regression classifier on the inner‑product of
    encoded (online, upstream) pair embeddings.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=2000)

    def fit(
        self,
        encoder: torch.nn.Module,
        tokenizer,
        D_R_train: List[Dict],
        D_PT: List[Dict],
        forgetting_labels: Dict[Tuple[int, int], int],
    ):
        X = []
        y = []
        # Pre‑compute encodings
        enc_cache_R = {i: encode_example(encoder, tokenizer, ex) for i, ex in enumerate(D_R_train)}
        enc_cache_PT = {j: encode_example(encoder, tokenizer, ex) for j, ex in enumerate(D_PT)}

        for (i, j), z in forgetting_labels.items():
            # Inner product
            vec = torch.nn.functional.cosine_similarity(
                enc_cache_R[i].unsqueeze(0), enc_cache_PT[j].unsqueeze(0)
            ).item()
            X.append([vec])
            y.append(z)

        self.model.fit(np.array(X), np.array(y))

    def predict(self, encoder, tokenizer, D_R_test: List[Dict], D_PT: List[Dict]) -> Dict[int, int]:
        preds = {j: 0 for j in range(len(D_PT))}
        # Compute encodings once
        enc_cache_R = {i: encode_example(encoder, tokenizer, ex) for i, ex in enumerate(D_R_test)}
        enc_cache_PT = {j: encode_example(encoder, tokenizer, ex) for j, ex in enumerate(D_PT)}
        for i, exR in enumerate(D_R_test):
            encR = enc_cache_R[i]
            for j, exP in enumerate(D_PT):
                encP = enc_cache_PT[j]
                vec = torch.nn.functional.cosine_similarity(encR.unsqueeze(0), encP.unsqueeze(0)).item()
                prob = self.model.predict_proba([[vec]])[0, 1]
                preds[j] = 1 if prob >= 0.5 else 0
        return preds


# --------------------------------------------------------------------------- #
# Logit‑change predictor (placeholder)
# --------------------------------------------------------------------------- #
def logit_change_placeholder(
    D_PT: List[Dict], D_R_train: List[Dict]
) -> Dict[int, int]:
    """
    Dummy implementation that returns random predictions.
    Included for completeness; not used in the main pipeline.
    """
    rng = np.random.RandomState(42)
    return {j: rng.randint(0, 2) for j in range(len(D_PT))}