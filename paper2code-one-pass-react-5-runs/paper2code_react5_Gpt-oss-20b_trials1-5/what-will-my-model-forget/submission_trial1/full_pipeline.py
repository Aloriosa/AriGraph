"""Orchestrates the full experimental pipeline."""
import json
import os
from data import load_small_agnews_split
from forecasting import (
    threshold_baseline,
    RepresentationForecaster,
)
from refinement import refine_and_evaluate
from evaluation import evaluate_forecasting

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_LENGTH = 128
K_STEPS = 30
LEARNING_RATE = 1e-4
THRESHOLD_GAMMA = 0.05  # Tuned manually for this toy setting

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
D_PT, D_R_train, D_R_test = load_small_agnews_split()

# --------------------------------------------------------------------------- #
# 1. Refine the model and collect forgetting labels
# --------------------------------------------------------------------------- #
edit_success, em_drop, forgetting_labels = refine_and_evaluate(
    D_PT, D_R_train, D_R_test, K=K_STEPS, lr=LEARNING_RATE, max_length=MAX_LENGTH
)

# --------------------------------------------------------------------------- #
# 2. Prepare ground‑truth forgetting labels for forecasting
# --------------------------------------------------------------------------- #
# The ground‑truth for forecasting is z_ij from the refinement step
# But we only need the labels for D_R_train (training) and D_R_test (testing)
# Build mappings: (i, j) -> label
# For evaluation, we need predictions for each j in D_PT given each i in D_R_test

# Build ground truth per j (average over i) for testing
ground_truth_test = {}
for j in range(len(D_PT)):
    # count how many times example j was forgotten across all test online examples
    count = sum(forgetting_labels[(i, j)] for i in range(len(D_R_test)))
    # For binary classification per example, we treat as 1 if ever forgotten
    ground_truth_test[j] = 1 if count > 0 else 0

# --------------------------------------------------------------------------- #
# 3. Threshold baseline
# --------------------------------------------------------------------------- #
# Compute frequencies on training data
freq_labels_train = {
    (i, j): forgetting_labels[(i, j)] for i in range(len(D_R_train)) for j in range(len(D_PT))
}
threshold_preds = threshold_baseline(
    D_PT, D_R_train, freq_labels_train, gamma=THRESHOLD_GAMMA
)

# --------------------------------------------------------------------------- #
# 4. Representation‑based forecaster
# --------------------------------------------------------------------------- #
# Load a fresh encoder for embeddings
from model_utils import load_base_model
model_enc, tokenizer_enc = load_base_model(max_length=MAX_LENGTH)
# Use the same model for encoding; we only need the encoder part
forecaster = RepresentationForecaster()
forecaster.fit(
    encoder=model_enc,
    tokenizer=tokenizer_enc,
    D_R_train=D_R_train,
    D_PT=D_PT,
    forgetting_labels=freq_labels_train,
)

# Predict on test set
rep_preds = forecaster.predict(model_enc, tokenizer_enc, D_R_test, D_PT)

# --------------------------------------------------------------------------- #
# 5. Evaluate forecasting
# --------------------------------------------------------------------------- #
results_forecasting = {}
results_forecasting["Threshold"] = evaluate_forecasting(threshold_preds, ground_truth_test)
results_forecasting["Representation"] = evaluate_forecasting(rep_preds, ground_truth_test)

# --------------------------------------------------------------------------- #
# 6. Save outputs
# --------------------------------------------------------------------------- #
os.makedirs("outputs", exist_ok=True)
with open("outputs/forecasting_results.json", "w") as f:
    json.dump(results_forecasting, f, indent=2)

refinement_results = {
    "Edit_Success_Rate_%": edit_success,
    "EM_Drop_Ratio_%": em_drop,
}
with open("outputs/refinement_results.json", "w") as f:
    json.dump(refinement_results, f, indent=2)

print("Reproduction finished. Results written to outputs/*.json")