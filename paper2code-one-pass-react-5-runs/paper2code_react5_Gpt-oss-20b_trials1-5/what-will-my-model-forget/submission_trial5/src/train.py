#!/usr/bin/env python
import json
import os
import random
from typing import List, Tuple

import torch
import numpy as np

from src.dataset import create_dataset
from src.model import BARTWrapper
from src.forecasting import ThresholdForecaster
from src.utils import (
    seed_everything,
    compute_em,
    compute_f1,
    compute_precision,
    compute_recall,
    save_json,
)

def main():
    seed_everything(12345)

    # 1. Create synthetic data
    pretrain_ds, online_ds = create_dataset(num_pretrain=50, num_online=5)

    # 2. Initialize model
    model = BARTWrapper("facebook/bart-base")

    # 3. Evaluate before any updates
    preds_before, golds_before = model.evaluate(pretrain_ds)
    em_before = compute_em(preds_before, golds_before)

    # 4. For each online example, fine‑tune model and record forgetting
    num_online = len(online_ds)
    num_pretrain = len(pretrain_ds)
    forgetting_matrix = np.zeros((num_online, num_pretrain), dtype=int)

    for i in range(num_online):
        inp = online_ds[i]["input_text"]
        tgt = online_ds[i]["target_text"]
        model.fine_tune_one_example(inp, tgt, lr=1e-4, num_steps=30)

        # Evaluate on pretrain set after update
        preds_after, _ = model.evaluate(pretrain_ds)
        em_after = compute_em(preds_after, golds_before)
        # Determine which pretrain examples became wrong
        for j, (pred, gold) in enumerate(zip(preds_after, golds_before)):
            if pred.strip() != gold.strip():
                forgetting_matrix[i, j] = 1

    em_after = compute_em(preds_after, golds_before)
    em_drop_ratio = (em_before - em_after) / em_before

    # 5. Forecasting
    forecaster = ThresholdForecaster(threshold=1)
    forecaster.fit(forgetting_matrix)
    # For evaluation we use the average forgetting mask across online examples
    # (i.e., an example is considered forgotten if it was forgotten for any online example)
    true_mask = (forgetting_matrix.sum(axis=0) > 0).astype(int)
    forecast_f1 = forecaster.evaluate(forgetting_matrix)

    # 6. Edit success rate (percentage of online examples that are correct after fine‑tune)
    preds_online, golds_online = model.evaluate(online_ds)
    edit_success = compute_em(preds_online, golds_online)

    results = {
        "EM_before": em_before,
        "EM_after": em_after,
        "EM_drop_ratio": em_drop_ratio,
        "Edit_Success_Rate": edit_success,
        "Forecast_F1": forecast_f1,
    }

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    save_json("results.json", results)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()