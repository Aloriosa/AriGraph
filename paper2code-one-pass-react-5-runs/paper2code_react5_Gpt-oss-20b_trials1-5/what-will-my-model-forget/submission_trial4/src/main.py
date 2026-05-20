# src/main.py
import os
import random
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from utils import (
    set_seed,
    load_model_and_tokenizer,
    encode_example,
    train_one_epoch,
    evaluate_example,
    compute_em,
    get_rep_vector,
    get_logit_target,
    load_p3_subset,
    load_mmlu_subset,
    EncoderMLP,
)

# --------------------------------------------------------------------------- #
#                          Hyper‑parameters                                 #
# --------------------------------------------------------------------------- #
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-large"      # or "google/flan-t5-large"
K_STEPS = 30
LR = 1e-5
BATCH_SIZE = 1
TOP_K_REPLAY = 5
REP_DIM = 128
GRAD_ACCUM = 1
NUM_UPSTREAM = 200
NUM_ONLINE = 10
NUM_TRAIN_ONLINE = 8

# --------------------------------------------------------------------------- #
#                          Helper functions                                 #
# --------------------------------------------------------------------------- #
def compute_forgetting_labels(baseline_correct, new_correct):
    """Return binary labels for forgetting (1 if correct before and wrong after)."""
    return [int(b and not n) for b, n in zip(baseline_correct, new_correct)]


def dot_product(a, b):
    return torch.dot(a, b).item()


# --------------------------------------------------------------------------- #
#                          Main experiment                                  #
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 1. Load data
    upstream_examples = load_p3_subset(num_examples=NUM_UPSTREAM, seed=SEED)
    online_examples = load_mmlu_subset(num_examples=NUM_ONLINE, seed=SEED)
    print(f"Upstream examples: {len(upstream_examples)}")
    print(f"Online examples: {len(online_examples)}")

    # 2. Split online examples into train/test
    train_online = online_examples[:NUM_TRAIN_ONLINE]
    test_online = online_examples[NUM_TRAIN_ONLINE:]

    # 3. Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, DEVICE)
    print(f"Loaded model {MODEL_NAME}")

    # 4. Baseline EM on upstream (before any updates)
    upstream_preds = []
    upstream_targets = [ex["output"] for ex in upstream_examples]
    for ex in tqdm(upstream_examples, desc="Baseline upstream inference"):
        pred, _, _ = evaluate_example(model, tokenizer, ex, DEVICE)
        upstream_preds.append(pred)
    baseline_em = compute_em(upstream_preds, upstream_targets)
    baseline_correct = [p == t for p, t in zip(upstream_preds, upstream_targets)]
    print(f"Baseline EM on upstream: {baseline_em:.4f}")

    # 5. Prepare encoder for representation methods
    encoder = EncoderMLP(hidden_dim=model.config.d_model, out_dim=REP_DIM)
    encoder.to(DEVICE)
    # Attach the model's encoder to our wrapper
    encoder.encoder = model.get_encoder()

    # 6. Prepare data structures for training forecasting models
    logit_features = []
    logit_labels = []
    repr_features = []
    repr_labels = []
    freq_counts = [0] * len(upstream_examples)

    # 7. Train forecasting models on training online examples
    print("\n=== Training forecasting models ===")
    for o_idx, online_ex in enumerate(tqdm(train_online, desc="Training online")):
        # Representations before update
        rep_online_before = get_rep_vector(model, tokenizer, online_ex, encoder, DEVICE)
        rep_up_before = [get_rep_vector(model, tokenizer, u_ex, encoder, DEVICE) for u_ex in upstream_examples]

        # Logits before update
        logit_online_before = get_logit_target(model, tokenizer, online_ex, DEVICE)
        logit_up_before = [get_logit_target(model, tokenizer, u_ex, DEVICE) for u_ex in upstream_examples]

        # Fine‑tune on the single online example
        online_loader = DataLoader([online_ex], batch_size=BATCH_SIZE, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_one_epoch(model, online_loader, optimizer, DEVICE, grad_accum=GRAD_ACCUM, num_steps=K_STEPS)

        # Representations after update
        rep_online_after = get_rep_vector(model, tokenizer, online_ex, encoder, DEVICE)
        rep_up_after = [get_rep_vector(model, tokenizer, u_ex, encoder, DEVICE) for u_ex in upstream_examples]

        # Logits after update
        logit_online_after = get_logit_target(model, tokenizer, online_ex, DEVICE)
        logit_up_after = [get_logit_target(model, tokenizer, u_ex, DEVICE) for u_ex in upstream_examples]

        # Delta representations and logits
        delta_rep_online = rep_online_after - rep_online_before
        delta_logit_online = logit_online_after - logit_online_before

        # New correctness on upstream after update
        new_preds = []
        for u_ex in upstream_examples:
            p, _, _ = evaluate_example(model, tokenizer, u_ex, DEVICE)
            new_preds.append(p)
        new_correct = [p == t for p, t in zip(new_preds, upstream_targets)]
        labels = compute_forgetting_labels(baseline_correct, new_correct)

        # Update frequency counts
        for j, l in enumerate(labels):
            if l == 1:
                freq_counts[j] += 1

        # Build features for training
        for j in range(len(upstream_examples)):
            # Logit‑change feature: similarity * delta_logit_norm
            similarity = dot_product(rep_up_before[j], rep_online_before)
            delta_norm = torch.norm(delta_logit_online).item()
            feat = [similarity * delta_norm]
            logit_features.append(feat)
            logit_labels.append(labels[j])

            # Representation‑based feature: inner product
            inner = dot_product(rep_up_before[j], rep_online_before)
            repr_features.append([inner])
            repr_labels.append(labels[j])

    # 8. Train forecasting models
    # Logit‑change model – simple MLP
    logit_clf = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    ).to(DEVICE)
    logit_optimizer = torch.optim.AdamW(logit_clf.parameters(), lr=LR)
    logit_criterion = torch.nn.BCELoss()

    # Representation‑based model – MLP with bias
    repr_clf = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    ).to(DEVICE)
    repr_optimizer = torch.optim.AdamW(repr_clf.parameters(), lr=LR)
    repr_criterion = torch.nn.BCELoss()

    # Convert training data to tensors
    logit_features_t = torch.tensor(logit_features, device=DEVICE, dtype=torch.float32)
    logit_labels_t = torch.tensor(logit_labels, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    repr_features_t = torch.tensor(repr_features, device=DEVICE, dtype=torch.float32)
    repr_labels_t = torch.tensor(repr_labels, device=DEVICE, dtype=torch.float32).unsqueeze(1)

    # Train for a few epochs
    for epoch in range(5):
        # Logit‑change
        logit_optimizer.zero_grad()
        logits = logit_clf(logit_features_t)
        loss = logit_criterion(logits, logit_labels_t)
        loss.backward()
        logit_optimizer.step()

        # Representation
        repr_optimizer.zero_grad()
        logits_r = repr_clf(repr_features_t)
        loss_r = repr_criterion(logits_r, repr_labels_t)
        loss_r.backward()
        repr_optimizer.step()

    print("Forecasting models trained.")

    # Frequency‑threshold baseline
    best_gamma = 0
    best_f1 = 0.0
    for gamma in range(max(freq_counts) + 1):
        preds = [1 if c >= gamma else 0 for c in freq_counts]
        f1 = f1_score(logit_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_gamma = gamma
    print(f"Frequency‑threshold baseline: gamma={best_gamma}, F1={best_f1:.4f}")

    # -----------------------------------------------------------------------
    # 9. Evaluate forecasting models on test set
    # -----------------------------------------------------------------------
    test_metrics = {
        "logit": {"f1": 0.0, "prec": 0.0, "rec": 0.0},
        "repr": {"f1": 0.0, "prec": 0.0, "rec": 0.0},
        "freq": {"f1": 0.0, "prec": 0.0, "rec": 0.0},
    }

    # Prepare frequency counts for test (use same counts as training)
    freq_counts_test = freq_counts  # same as training

    # Reset model to original weights before test evaluation
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, DEVICE)

    print("\n=== Evaluating on test online examples ===")
    for o_idx, online_ex in enumerate(tqdm(test_online, desc="Testing online")):
        # Representations before update
        rep_online_before = get_rep_vector(model, tokenizer, online_ex, encoder, DEVICE)
        rep_up_before = [get_rep_vector(model, tokenizer, u_ex, encoder, DEVICE) for u_ex in upstream_examples]

        # Logits before update
        logit_online_before = get_logit_target(model, tokenizer, online_ex, DEVICE)
        logit_up_before = [get_logit_target(model, tokenizer, u_ex, DEVICE) for u_ex in upstream_examples]

        # Fine‑tune on the single online example
        online_loader = DataLoader([online_ex], batch_size=BATCH_SIZE, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_one_epoch(model, online_loader, optimizer, DEVICE, grad_accum=GRAD_ACCUM, num_steps=K_STEPS)

        # Representations after update
        rep_online_after = get_rep_vector(model, tokenizer, online_ex, encoder, DEVICE)
        rep_up_after = [get_rep_vector(model, tokenizer, u_ex, encoder, DEVICE) for u_ex in upstream_examples]

        # Logits after update
        logit_online_after = get_logit_target(model, tokenizer, online_ex, DEVICE)
        logit_up_after = [get_logit_target(model, tokenizer, u_ex, DEVICE) for u_ex in upstream_examples]

        # Ground truth forgetting labels
        new_preds = []
        for u_ex in upstream_examples:
            p, _, _ = evaluate_example(model, tokenizer, u_ex, DEVICE)
            new_preds.append(p)
        new_correct = [p == t for p, t in zip(new_preds, upstream_targets)]
        true_labels = compute_forgetting_labels(baseline_correct, new_correct)

        # Prediction with each model
        preds_logit = []
        preds_repr = []
        preds_freq = []

        for j in range(len(upstream_examples)):
            # Logit‑change prediction
            similarity = dot_product(rep_up_before[j], rep_online_before)
            delta_norm = torch.norm(logit_online_after - logit_online_before).item()
            feat = torch.tensor([[similarity * delta_norm]], device=DEVICE)
            prob = logit_clf(feat).item()
            preds_logit.append(1 if prob >= 0.5 else 0)

            # Representation‑based prediction
            inner = dot_product(rep_up_before[j], rep_online_before)
            prob2 = repr_clf(torch.tensor([[inner]], device=DEVICE)).item()
            preds_repr.append(1 if prob2 >= 0.5 else 0)

            # Frequency baseline
            preds_freq.append(1 if freq_counts_test[j] >= best_gamma else 0)

        # Compute metrics
        for name, preds in [("logit", preds_logit), ("repr", preds_repr), ("freq", preds_freq)]:
            f1 = f1_score(true_labels, preds)
            prec = precision_score(true_labels, preds, zero_division=0)
            rec = recall_score(true_labels, preds, zero_division=0)
            test_metrics[name]["f1"] += f1
            test_metrics[name]["prec"] += prec
            test_metrics[name]["rec"] += rec

        # ==================== Replay ======================
        # Random replay baseline
        random_indices = random.sample(range(len(upstream_examples)), TOP_K_REPLAY)
        random_set = [upstream_examples[i] for i in random_indices]

        # Representation‑based replay
        probs = []
        for j in range(len(upstream_examples)):
            inner = dot_product(rep_up_before[j], rep_online_before)
            prob = repr_clf(torch.tensor([[inner]], device=DEVICE)).item()
            probs.append((prob, j))
        probs.sort(reverse=True, key=lambda x: x[0])
        repr_topk = [upstream_examples[j] for _, j in probs[:TOP_K_REPLAY]]

        # Fine‑tune on online + random
        random_loader = DataLoader([online_ex] + random_set, batch_size=BATCH_SIZE, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_one_epoch(model, random_loader, optimizer, DEVICE, grad_accum=GRAD_ACCUM, num_steps=K_STEPS)

        # Edit success after random replay
        _, _, succ_random = evaluate_example(model, tokenizer, online_ex, DEVICE)
        # EM on upstream after random replay
        preds_after_random = [
            evaluate_example(model, tokenizer, u_ex, DEVICE)[0] for u_ex in upstream_examples
        ]
        em_after_random = compute_em(preds_after_random, upstream_targets)
        em_drop_rand = (baseline_em - em_after_random) / baseline_em * 100.0

        # Fine‑tune on online + repr‑based
        repr_loader = DataLoader([online_ex] + repr_topk, batch_size=BATCH_SIZE, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_one_epoch(model, repr_loader, optimizer, DEVICE, grad_accum=GRAD_ACCUM, num_steps=K_STEPS)

        # Edit success after repr replay
        _, _, succ_repr = evaluate_example(model, tokenizer, online_ex, DEVICE)
        # EM on upstream after repr replay
        preds_after_repr = [
            evaluate_example(model, tokenizer, u_ex, DEVICE)[0] for u_ex in upstream_examples
        ]
        em_after_repr = compute_em(preds_after_repr, upstream_targets)
        em_drop_repr = (baseline_em - em_after_repr) / baseline_em * 100.0

        # Store results for averaging
        if o_idx == 0:
            replay_results = {
                "edit_success_rand": [],
                "edit_success_repr": [],
                "em_drop_rand": [],
                "em_drop_repr": [],
            }
        replay_results["edit_success_rand"].append(succ_random)
        replay_results["edit_success_repr"].append(succ_repr)
        replay_results["em_drop_rand"].append(em_drop_rand)
        replay_results["em_drop_repr"].append(em_drop_repr)

    # Average metrics
    n = len(test_online)
    for name in test_metrics:
        test_metrics[name]["f1"] /= n
        test_metrics[name]["prec"] /= n
        test_metrics[name]["rec"] /= n

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    print("\n=== Forecasting Model Evaluation on Test Set ===")
    for name in test_metrics:
        print(f"{name.capitalize()} model: F1={test_metrics[name]['f1']:.4f} "
              f"Prec={test_metrics[name]['prec']:.4f} Rec={test_metrics[name]['rec']:.4f}")

    avg_edit_success_repr = np.mean(replay_results["edit_success_repr"])
    avg_edit_success_rand = np.mean(replay_results["edit_success_rand"])
    avg_em_drop_repr = np.mean(replay_results["em_drop_repr"])
    avg_em_drop_rand = np.mean(replay_results["em_drop_rand"])
    print("\n=== Replay Results ===")
    print(f"Average Edit Success (Random replay): {avg_edit_success_rand*100:.2f}%")
    print(f"Average Edit Success (Representation replay): {avg_edit_success_repr*100:.2f}%")
    print(f"Average EM Drop (Random replay): {avg_em_drop_rand:.2f}%")
    print(f"Average EM Drop (Representation replay): {avg_em_drop_repr:.2f}%")

    # Clean up large tensors to keep repo clean
    del model, tokenizer, encoder, upstream_examples, online_examples
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()