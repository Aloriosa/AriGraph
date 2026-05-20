import numpy as np
from typing import List, Tuple

class ThresholdForecaster:
    """
    Predict forgotten examples based on a frequency threshold.
    The frequency of forgetting is computed on the training set.
    """
    def __init__(self, threshold: int = 1):
        self.threshold = threshold
        self.forgotten_counts = None

    def fit(self, forgotten_matrix: np.ndarray):
        """
        forgotten_matrix: shape (num_online, num_pretrain)
        1 indicates the pretrain example was forgotten when fixing that online example.
        """
        # Count how many times each pretrain example was forgotten across all online examples
        self.forgotten_counts = forgotten_matrix.sum(axis=0)

    def predict(self) -> np.ndarray:
        """
        Return a binary mask of pretrain examples predicted to be forgotten.
        """
        return (self.forgotten_counts >= self.threshold).astype(int)

    def evaluate(self, true_mask: np.ndarray) -> float:
        """
        Compute F1 between predicted mask and true mask (average over all online examples).
        """
        preds = self.predict()
        return np.mean(
            [
                self._binary_f1(preds, true_mask[i])
                for i in range(true_mask.shape[0])
            ]
        )

    @staticmethod
    def _binary_f1(pred: np.ndarray, truth: np.ndarray) -> float:
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        if tp + fp + fn == 0:
            return 1.0
        return 2 * tp / (2 * tp + fp + fn)