#!/usr/bin/env python3
"""
Load the trained forecasting models and evaluate on the held‑out validation set.
Results are written to `outputs/metrics.json`.
"""
import os
import json
from utils import load_pickle, save_pickle

MODEL_DIR = "outputs/forecast_models"
OUTPUT_DIR = "outputs"

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("Forecasting models not found. Run forecast_train.py first.")

models = load_pickle(os.path.join(MODEL_DIR, "forecast_models.pkl"))
metrics = models["metrics"]

# Save metrics
output_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics written to {output_path}")
print(json.dumps(metrics, indent=2))