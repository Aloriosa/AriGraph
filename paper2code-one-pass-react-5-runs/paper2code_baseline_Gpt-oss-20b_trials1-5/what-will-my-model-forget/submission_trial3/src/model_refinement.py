# src/model_refinement.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ModelRefinement:
    """
    Handles fine‑tuning a pretrained model on single online examples
    and computing ground‑truth forgetting for PT examples.
    """
    def __init__(self, model_name='facebook/bart-base', device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)
        self.model.eval()

    def _predict(self, example):
        """
        example: dict with 'sentence' and 'label'
        Returns logits (torch.tensor) and predicted label (int).
        """
        inputs = self.tokenizer(
            example['sentence'],
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)
        pred = int(torch.argmax(logits).item())
        return logits, pred

    def compute_ground_truth_forgetting(
        self, pt_examples, online_example, update_steps=5, lr=1e-5
    ):
        """
        Fine‑tune the model on a single online example for `update_steps`.
        Return a list of tuples:
          (pt_idx, GroundTruthLabel, logits_before, logits_after, is_forgotten)
        """
        # cache logits before update
        logits_before = []
        preds_before = []
        for ex in pt_examples:
            logit, pred = self._predict(ex)
            logits_before.append(logit.cpu())
            preds_before.append(pred)

        # fine‑tune
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for _ in range(update_steps):
            inputs = self.tokenizer(
                online_example['sentence'],
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=128
            ).to(self.device)
            labels = torch.tensor([online_example['label']], device=self.device)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.model.eval()

        # cache logits after update
        logits_after = []
        preds_after = []
        for ex in pt_examples:
            logit, pred = self._predict(ex)
            logits_after.append(logit.cpu())
            preds_after.append(pred)

        # compute forgetting
        results = []
        for idx, (logit_b, pred_b, logit_a, pred_a, ex) in enumerate(
            zip(logits_before, preds_before, logits_after, preds_after, pt_examples)
        ):
            is_forgotten = (pred_b == ex['label']) and (pred_a != ex['label'])
            results.append((idx, is_forgotten, logit_b, logit_a, ex))
        return results

    def get_hidden_representation(self, example):
        """
        Returns the CLS token representation of the encoder.
        """
        inputs = self.tokenizer(
            example['sentence'],
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.bert(**inputs)  # encoder outputs
        # CLS token is the first token
        cls = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
        return cls.squeeze(0)  # [hidden_dim]