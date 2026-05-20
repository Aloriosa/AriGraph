"""Model refinement logic: fine‑tune on online errors and compute metrics."""
import torch
from typing import List, Dict, Tuple
from model_utils import fine_tune_one_example, load_base_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset: List[Dict],
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return predictions and ground truths for a dataset.
    """
    preds = []
    gts = []
    for ex in dataset:
        input_text = f"Classify this news: {ex['text']}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                do_sample=False,
                num_beams=1,
            )
        pred_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            pred = int(pred_str.strip())
        except:
            pred = -1  # invalid prediction
        preds.append(pred)
        gts.append(ex["label"])
    return np.array(preds), np.array(gts)


def compute_metrics(preds, gts) -> Dict[str, float]:
    """Compute exact match accuracy (EM)."""
    correct = (preds == gts).sum()
    return {"EM": correct / len(gts) * 100.0}


def refine_and_evaluate(
    D_PT: List[Dict],
    D_R_train: List[Dict],
    D_R_test: List[Dict],
    K: int = 30,
    lr: float = 1e-4,
    max_length: int = 128,
):
    """
    For each online example in D_R_test:
        * Fine‑tune the base model for K steps.
        * Compute edit success (whether the online example is now correct).
        * Compute EM on D_PT before and after, to derive EM Drop Ratio.
        * Record forgetting labels z_ij for all D_PT examples.
    Returns:
        - edit_success_rate (float)
        - em_drop_ratio (float)
        - forgetting_labels: dict {(i, j): 0/1}
    """
    base_model, tokenizer = load_base_model(device=DEVICE)
    base_model.eval()

    # EM before any updates
    _, gts_PT = evaluate_on_dataset(base_model, tokenizer, D_PT, max_length)
    em_before = compute_metrics(*evaluate_on_dataset(base_model, tokenizer, D_PT, max_length))["EM"]

    # Prepare storage
    correct_online = 0
    forgetting_labels = {}  # (i, j) -> 0/1

    # For each online example in test set
    for i, ex_online in enumerate(D_R_test):
        # Fine‑tune on this single example
        updated_model = fine_tune_one_example(
            base_model,
            tokenizer,
            ex_online,
            K=K,
            lr=lr,
            max_length=max_length,
        )

        # Edit success on this online example
        pred_online, _ = evaluate_on_dataset(updated_model, tokenizer, [ex_online], max_length)
        if pred_online[0] == ex_online["label"]:
            correct_online += 1

        # EM on D_PT after update
        _, gts_PT_after = evaluate_on_dataset(updated_model, tokenizer, D_PT, max_length)
        em_after = compute_metrics(*evaluate_on_dataset(updated_model, tokenizer, D_PT, max_length))["EM"]

        # EM Drop Ratio for this step (we will average later)
        # Record forgetting for each D_PT example
        for j, ex_PT in enumerate(D_PT):
            # Was correct before?
            correct_before = (pred_online[0] == ex_online["label"])  # placeholder
            # Determine if the PT example changed from correct to incorrect
            # We need the prediction of the PT example before update
            pred_PT_before, _ = evaluate_on_dataset(base_model, tokenizer, [ex_PT], max_length)
            pred_PT_after, _ = evaluate_on_dataset(updated_model, tokenizer, [ex_PT], max_length)
            forgotten = int((pred_PT_before[0] == ex_PT["label"]) and (pred_PT_after[0] != ex_PT["label"]))
            forgetting_labels[(i, j)] = forgotten

        # Update base model reference for next iteration
        base_model = updated_model

    edit_success_rate = correct_online / len(D_R_test) * 100.0

    # Re‑compute EM after all updates for final EM Drop Ratio
    _, gts_PT_final = evaluate_on_dataset(base_model, tokenizer, D_PT, max_length)
    em_final = compute_metrics(*evaluate_on_dataset(base_model, tokenizer, D_PT, max_length))["EM"]
    em_drop_ratio = (em_before - em_final) / em_before * 100.0 if em_before > 0 else 0.0

    return edit_success_rate, em_drop_ratio, forgetting_labels