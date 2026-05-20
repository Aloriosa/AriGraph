#!/usr/bin/env python
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from utils import (
    evaluate_em,
    get_error_examples,
    get_pretraining_examples,
    load_json,
    load_squad,
    save_json,
)
from forecast_utils import (
    get_example_repr,
    train_logistic_regression,
    predict_forgetting,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-base"
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_TRAIN_STEPS = 30
NUM_ERRORS = 20
NUM_PT_SAMPLES = 100
TOP_K_REPLAY = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main():
    # 1. Load data
    squad_train, squad_dev = load_squad()

    # 2. Prepare tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 3. Build D_PT
    d_pt = get_pretraining_examples(squad_train, num_samples=NUM_PT_SAMPLES, seed=SEED)
    logger.info(f"D_PT size: {len(d_pt)}")

    # 4. Build D_R (error examples)
    d_r = get_error_examples(squad_dev, base_model, tokenizer, num_errors=NUM_ERRORS, seed=SEED)
    logger.info(f"D_R (errors) size: {len(d_r)}")

    # 5. Baseline EM on D_PT
    baseline_em = evaluate_em(base_model, tokenizer, d_pt, batch_size=BATCH_SIZE, max_length=MAX_LEN)
    logger.info(f"Baseline EM on D_PT: {baseline_em:.2f}%")

    # 6. Pre‑compute representations of D_PT using base model
    pt_reprs = [get_example_repr(base_model, tokenizer, ex, max_length=MAX_LEN) for ex in d_pt]

    # 7. Sequentially fine‑tune on errors and record forgetting
    # We keep a copy of the base model for each error to avoid cumulative drift
    forgetting_records = []  # list of tuples (error_index, pt_index, label)
    for idx, err_ex in enumerate(d_r):
        logger.info(f"Fine‑tuning on error {idx+1}/{len(d_r)}")

        # Create a trainer that fine‑tunes on a single example
        train_args = Seq2SeqTrainingArguments(
            output_dir=f"tmp_{idx}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=LEARNING_RATE,
            num_train_epochs=1,
            logging_steps=10,
            do_train=True,
            do_eval=False,
            save_strategy="no",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            seed=SEED,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)

        # Prepare single‑example dataset
        train_dataset = Dataset.from_dict(
            {
                "input_ids": [tokenizer(
                    f"question: {err_ex['question']} context: {err_ex['context']}",
                    max_length=MAX_LEN,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).input_ids.squeeze(0)],
                "labels": [tokenizer(
                    err_ex["answers"]["text"][0] if err_ex["answers"]["text"] else "",
                    max_length=MAX_LEN,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).input_ids.squeeze(0)],
            }
        )

        # Clone base model for this error
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        # 7a. Evaluate forgetting on D_PT
        model.eval()
        for j, pt_ex in enumerate(d_pt):
            # Check if base model was correct
            inp = f"question: {pt_ex['question']} context: {pt_ex['context']}"
            input_ids = tokenizer(inp, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length").input_ids
            input_ids = input_ids.to(DEVICE)
            with torch.no_grad():
                out_ids = model.generate(input_ids, max_length=MAX_LEN, num_beams=4, early_stopping=True)
            pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            gold = pt_ex["answers"]["text"][0] if pt_ex["answers"]["text"] else ""
            base_correct = (pred.strip() == gold.strip())

            # Get updated prediction
            with torch.no_grad():
                out_ids_new = trainer.model.generate(input_ids, max_length=MAX_LEN, num_beams=4, early_stopping=True)
            pred_new = tokenizer.decode(out_ids_new[0], skip_special_tokens=True)
            new_correct = (pred_new.strip() == gold.strip())

            # Label: 1 if forgetting (was correct, now incorrect)
            label = 1 if base_correct and not new_correct else 0
            forgetting_records.append((idx, j, label))

    # 8. Train representation‑based classifier
    logger.info("Training forecasting classifier...")
    error_reprs = []
    pt_reprs_all = []
    labels = []
    for err_idx, pt_idx, lbl in forgetting_records:
        err_ex = d_r[err_idx]
        pt_ex = d_pt[pt_idx]
        er = get_example_repr(base_model, tokenizer, err_ex, max_length=MAX_LEN)
        pr = pt_reprs[pt_idx]
        error_reprs.append(er)
        pt_reprs_all.append(pr)
        labels.append(lbl)

    clf = train_logistic_regression(error_reprs, pt_reprs_all, labels)
    logger.info("Classifier trained.")

    # 9. Replay predicted forgotten examples
    # For each error, predict forgotten PT examples and fine‑tune on top_k
    logger.info("Replay phase with predicted forgotten examples...")
    model_for_replay = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    for err_idx, err_ex in enumerate(d_r):
        er = get_example_repr(base_model, tokenizer, err_ex, max_length=MAX_LEN)
        pred_idxs = predict_forgetting(clf, er, pt_reprs, threshold=0.5)
        # Keep only top_k
        topk = pred_idxs[:TOP_K_REPLAY]
        if not topk:
            continue

        # Build dataset of topk examples
        train_samples = []
        for pk in topk:
            ex = d_pt[pk]
            train_samples.append(
                {
                    "input_ids": tokenizer(
                        f"question: {ex['question']} context: {ex['context']}",
                        max_length=MAX_LEN,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).input_ids.squeeze(0),
                    "labels": tokenizer(
                        ex["answers"]["text"][0] if ex["answers"]["text"] else "",
                        max_length=MAX_LEN,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).input_ids.squeeze(0),
                }
            )
        # Fine‑tune for a few steps
        train_dataset = Dataset.from_dict(
            {
                "input_ids": [s["input_ids"] for s in train_samples],
                "labels": [s["labels"] for s in train_samples],
            }
        )
        train_args = Seq2SeqTrainingArguments(
            output_dir=f"replay_{err_idx}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=LEARNING_RATE,
            num_train_epochs=1,
            logging_steps=10,
            do_train=True,
            do_eval=False,
            save_strategy="no",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            seed=SEED,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_for_replay)
        trainer = Seq2SeqTrainer(
            model=model_for_replay,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

    # 10. Evaluate EM drop after replay
    final_em = evaluate_em(model_for_replay, tokenizer, d_pt, batch_size=BATCH_SIZE, max_length=MAX_LEN)
    em_drop = baseline_em - final_em
    logger.info(f"EM after replaying predicted forgotten examples: {final_em:.2f}%")
    logger.info(f"EM drop reduced from {baseline_em - final_em:.2f}% to {em_drop:.2f}%")

    # Save results
    results = {
        "baseline_em": baseline_em,
        "final_em": final_em,
        "em_drop": em_drop,
    }
    Path("results.json").write_text(str(results))
    logger.info("Results written to results.json")


if __name__ == "__main__":
    main()