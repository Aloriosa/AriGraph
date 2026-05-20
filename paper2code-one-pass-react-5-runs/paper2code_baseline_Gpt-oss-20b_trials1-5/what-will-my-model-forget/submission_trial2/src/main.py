import json
import pickle
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils import set_seed, em_score, compute_f1, compute_precision, compute_recall
from src.forecasting import ThresholdForecaster, LogitForecaster, RepresentationForecaster

# ------------------
# Configuration
# ------------------
SEED = 42
torch.set_grad_enabled(True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# Use small dataset for quick demo
DATASET_NAME = "ag_news"
MAX_TOKENS = 512
BATCH_SIZE = 4
NUM_TRAIN_ERRORS = 10   # number of online errors used for training forecasting
NUM_TEST_ERRORS = 5     # errors used for evaluation
NUM_PT_EXAMPLES = 100   # upstream pretrain examples

# ------------------
# Data preparation
# ------------------
print("[+] Loading dataset...")
raw_ds = load_dataset(DATASET_NAME, split="train[:2000]")  # small subset
raw_ds = raw_ds.shuffle(seed=SEED).train_test_split(test_size=0.1, seed=SEED)
train_ds = raw_ds['train']
test_ds = raw_ds['test']

# Prepare tokenizer and model
print("[+] Loading model and tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)
model.eval()

# Helper to encode a batch of examples
def encode_examples(examples):
    inputs = [
        f"agnews: {ex['text']}" for ex in examples
    ]
    targets = [ex['label'] for ex in examples]  # labels are integers 0-3
    target_texts = [tokenizer.convert_ids_to_tokens([t])[0] for t in targets]  # simple mapping
    target_texts = [
        ["label_0","label_1","label_2","label_3"][t] for t in targets
    ]
    enc = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt"
    )
    dec = tokenizer(
        target_texts,
        padding='max_length',
        truncation=True,
        max_length=10,
        return_tensors="pt"
    )
    return enc, dec

# ------------------
# Build upstream pre‑training set (D_PT)
# ------------------
print("[+] Building upstream pre‑training set D_PT ({} examples)...".format(NUM_PT_EXAMPLES))
pt_examples = train_ds.shuffle(seed=SEED).select(range(NUM_PT_EXAMPLES))
pt_inputs, pt_targets = encode_examples(pt_examples)
pt_inputs = {k: v.to(DEVICE) for k, v in pt_inputs.items()}
pt_targets = {k: v.to(DEVICE) for k, v in pt_targets.items()}

# ------------------
# Build online error sets
# ------------------
print("[+] Building online error sets...")
train_errors = test_ds.shuffle(seed=SEED).select(range(NUM_TRAIN_ERRORS))
test_errors = test_ds.shuffle(seed=SEED).select(range(NUM_TRAIN_ERRORS, NUM_TRAIN_ERRORS+NUM_TEST_ERRORS))

# ------------------
# Helper: fine‑tune on a single example
# ------------------
def fine_tune_single(example, steps=5, lr=5e-5, weight_decay=0.0):
    """Fine‑tune the base model on a single example for a few steps."""
    model_copy = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)
    model_copy.train()
    optimizer = AdamW(model_copy.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    enc, dec = encode_examples([example])
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    dec = {k: v.to(DEVICE) for k, v in dec.items()}

    for step in range(total_steps):
        outputs = model_copy(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            labels=dec['input_ids'],
            decoder_attention_mask=dec['attention_mask'],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return model_copy

# ------------------
# Compute forgetting labels for a given online error
# ------------------
def compute_forgetting_label(model_base, model_updated, pt_inputs, pt_targets):
    """
    Returns a binary array of length len(pt_inputs) indicating whether each
    upstream example is forgotten (prediction changed from correct to incorrect).
    """
    model_base.eval()
    model_updated.eval()
    # Base predictions
    with torch.no_grad():
        outputs_base = model_base.generate(
            input_ids=pt_inputs['input_ids'],
            attention_mask=pt_inputs['attention_mask'],
            max_new_tokens=8
        )
    pred_base = tokenizer.batch_decode(outputs_base, skip_special_tokens=True)

    # Updated predictions
    with torch.no_grad():
        outputs_upd = model_updated.generate(
            input_ids=pt_inputs['input_ids'],
            attention_mask=pt_inputs['attention_mask'],
            max_new_tokens=8
        )
    pred_upd = tokenizer.batch_decode(outputs_upd, skip_special_tokens=True)

    # Ground truth labels
    true_labels = [tokenizer.convert_ids_to_tokens([t])[0] for t in pt_targets['input_ids'].tolist()]
    true_labels = ["label_0","label_1","label_2","label_3"]  # mapping
    true_labels = [true_labels[t] for t in pt_targets['input_ids'].squeeze(1).tolist()]

    # Compute exact match before and after
    old_em = em_score(pred_base, true_labels)
    new_em = em_score(pred_upd, true_labels)

    # Forgetting: previously correct but now incorrect
    forgetting = []
    for old, new, true in zip(pred_base, pred_upd, true_labels):
        old_ok = (old.strip() == true.strip())
        new_ok = (new.strip() == true.strip())
        forgetting.append(1 if old_ok and not new_ok else 0)
    return np.array(forgetting, dtype=int)

# ------------------
# Build training data for forecasting models
# ------------------
print("[+] Building training data for forecasting...")
train_matrix = []  # shape (len(train_errors), len(pt_examples))
for ex in tqdm(train_errors, desc="Training errors"):
    model_upd = fine_tune_single(ex, steps=5, lr=5e-5)
    forgetting = compute_forgetting_label(model, model_upd, pt_inputs, pt_targets)
    train_matrix.append(forgetting)
train_matrix = np.array(train_matrix)  # shape (num_train_errors, num_pt_examples)

# ------------------
# Train threshold forecaster
# ------------------
print("[+] Training threshold forecaster...")
threshold_forecaster = ThresholdForecaster(gamma=0.2)
threshold_forecaster.fit(list(range(NUM_PT_EXAMPLES)), train_matrix)

# ------------------
# Train logit‑based forecaster
# ------------------
print("[+] Training logit‑based forecaster...")
logit_forecaster = LogitForecaster()
logit_forecaster.to(DEVICE)
optimizer_logit = AdamW(logit_forecaster.parameters(), lr=5e-5)
criterion_logit = nn.MarginRankingLoss(margin=1.0)

# We use a very small batch: one pair (online, pretrain) per step
for epoch in range(3):
    for i, ex in enumerate(tqdm(train_errors, desc=f"Epoch {epoch+1}")):
        model_upd = fine_tune_single(ex, steps=2, lr=5e-5)
        # Get hidden states for online example
        with torch.no_grad():
            inputs, _ = encode_examples([ex])
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model.base_model.encoder(**inputs)
            online_repr = logit_forecaster.encode(outputs.last_hidden_state)  # [1,64]
            # Get logits for online example before and after
            logits_before = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8,
                return_dict=True,
                output_scores=True
            ).scores[-1]  # last token logits
            logits_after = model_upd.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8,
                return_dict=True,
                output_scores=True
            ).scores[-1]

        # Pick a random pretrain example
        j = random.randint(0, NUM_PT_EXAMPLES-1)
        # Get its hidden states
        with torch.no_grad():
            pt_input = {
                'input_ids': pt_inputs['input_ids'][j:j+1],
                'attention_mask': pt_inputs['attention_mask'][j:j+1]
            }
            pt_out = model.base_model.encoder(**pt_input)
            pt_repr = logit_forecaster.encode(pt_out.last_hidden_state)
            pt_logits_before = model.generate(
                input_ids=pt_input['input_ids'],
                attention_mask=pt_input['attention_mask'],
                max_new_tokens=8,
                return_dict=True,
                output_scores=True
            ).scores[-1]

        # Predict logits for pretrain example
        pred_logits = logit_forecaster.predict_logit(
            online_repr, pt_repr, logits_after, pt_logits_before
        )

        # Compute loss: margin ranking on correct token
        # For simplicity we compare argmax scores
        pred_label = torch.argmax(pred_logits, dim=-1)
        true_label = torch.argmax(pt_logits_before, dim=-1)
        loss = torch.mean(F.mse_loss(pred_logits, pt_logits_before))
        optimizer_logit.zero_grad()
        loss.backward()
        optimizer_logit.step()

# ------------------
# Train representation‑based forecaster
# ------------------
print("[+] Training representation‑based forecaster...")
rep_forecaster = RepresentationForecaster()
rep_forecaster.to(DEVICE)
optimizer_rep = AdamW(rep_forecaster.parameters(), lr=5e-5)
criterion_rep = nn.BCEWithLogitsLoss()

biases = torch.zeros(NUM_PT_EXAMPLES, device=DEVICE)  # frequency prior
for epoch in range(3):
    for i, ex in enumerate(tqdm(train_errors, desc=f"Epoch {epoch+1}")):
        model_upd = fine_tune_single(ex, steps=2, lr=5e-5)
        # Encode online example
        with torch.no_grad():
            inputs, _ = encode_examples([ex])
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            online_repr = rep_forecaster.encode(model.base_model.encoder(**inputs).last_hidden_state)
        # Random pretrain example
        j = random.randint(0, NUM_PT_EXAMPLES-1)
        with torch.no_grad():
            pt_input = {
                'input_ids': pt_inputs['input_ids'][j:j+1],
                'attention_mask': pt_inputs['attention_mask'][j:j+1]
            }
            pt_repr = rep_forecaster.encode(model.base_model.encoder(**pt_input).last_hidden_state)
        # Get label from training matrix
        label = torch.tensor(train_matrix[i, j], dtype=torch.float32, device=DEVICE)
        logits = rep_forecaster(online_repr, pt_repr, biases[j].unsqueeze(0))
        loss = criterion_rep(logits, label)
        optimizer_rep.zero_grad()
        loss.backward()
        optimizer_rep.step()

# ------------------
# Evaluation on test errors
# ------------------
print("[+] Evaluating forecasting on test errors...")
test_matrix = []
for ex in tqdm(test_errors, desc="Test errors"):
    model_upd = fine_tune_single(ex, steps=5, lr=5e-5)
    forgetting = compute_forgetting_label(model, model_upd, pt_inputs, pt_targets)
    test_matrix.append(forgetting)
test_matrix = np.array(test_matrix)

# Threshold predictions
threshold_preds = threshold_forecaster.predict(np.arange(NUM_PT_EXAMPLES))
threshold_f1 = compute_f1(threshold_preds, test_matrix.mean(axis=0).round().astype(int))
threshold_prec = compute_precision(threshold_preds, test_matrix.mean(axis=0).round().astype(int))
threshold_rec = compute_recall(threshold_preds, test_matrix.mean(axis=0).round().astype(int))

# Logit predictions
logit_preds = []
for ex in test_errors:
    model_upd = fine_tune_single(ex, steps=2, lr=5e-5)
    # encode online
    with torch.no_grad():
        inputs, _ = encode_examples([ex])
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        online_repr = logit_forecaster.encode(model.base_model.encoder(**inputs).last_hidden_state)
        logits_before = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=8,
            return_dict=True,
            output_scores=True
        ).scores[-1]
        logits_after = model_upd.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=8,
            return_dict=True,
            output_scores=True
        ).scores[-1]
    preds_for_j = []
    for j in range(NUM_PT_EXAMPLES):
        with torch.no_grad():
            pt_input = {
                'input_ids': pt_inputs['input_ids'][j:j+1],
                'attention_mask': pt_inputs['attention_mask'][j:j+1]
            }
            pt_repr = logit_forecaster.encode(model.base_model.encoder(**pt_input).last_hidden_state)
            pt_logits_before = model.generate(
                input_ids=pt_input['input_ids'],
                attention_mask=pt_input['attention_mask'],
                max_new_tokens=8,
                return_dict=True,
                output_scores=True
            ).scores[-1]
            pred_logits = logit_forecaster.predict_logit(
                online_repr, pt_repr, logits_after, pt_logits_before
            )
            pred_label = torch.argmax(pred_logits, dim=-1)
            true_label = torch.argmax(pt_logits_before, dim=-1)
            preds_for_j.append(1 if pred_label != true_label else 0)
    logit_preds.append(preds_for_j)
logit_preds = np.array(logit_preds)
logit_f1 = compute_f1(logit_preds, test_matrix)
logit_prec = compute_precision(logit_preds, test_matrix)
logit_rec = compute_recall(logit_preds, test_matrix)

# Representation predictions
rep_preds = []
for ex in test_errors:
    with torch.no_grad():
        inputs, _ = encode_examples([ex])
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        online_repr = rep_forecaster.encode(model.base_model.encoder(**inputs).last_hidden_state)
    preds_for_j = []
    for j in range(NUM_PT_EXAMPLES):
        with torch.no_grad():
            pt_input = {
                'input_ids': pt_inputs['input_ids'][j:j+1],
                'attention_mask': pt_inputs['attention_mask'][j:j+1]
            }
            pt_repr = rep_forecaster.encode(model.base_model.encoder(**pt_input).last_hidden_state)
            logits = rep_forecaster(online_repr, pt_repr, biases[j].unsqueeze(0))
            pred = torch.sigmoid(logits).item() > 0.5
            preds_for_j.append(1 if pred else 0)
    rep_preds.append(preds_for_j)
rep_preds = np.array(rep_preds)
rep_f1 = compute_f1(rep_preds, test_matrix)
rep_prec = compute_precision(rep_preds, test_matrix)
rep_rec = compute_recall(rep_preds, test_matrix)

metrics = {
    "threshold_f1": threshold_f1, "threshold_prec": threshold_prec, "threshold_rec": threshold_rec,
    "logit_f1": logit_f1, "logit_prec": logit_prec, "logit_rec": logit_rec,
    "rep_f1": rep_f1, "rep_prec": rep_prec, "rep_rec": rep_rec,
}

print("[+] Forecasting results:")
print(json.dumps(metrics, indent=2))

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# ------------------
# Replay‑based refinement loop
# ------------------
print("[+] Running replay‑based refinement loop...")
# We'll replay the top‑k predicted forgotten examples for each error
k_replay = 5
edit_success = []
em_drop_list = []

# Compute baseline EM on D_PT before any updates
with torch.no_grad():
    baseline_outputs = model.generate(
        input_ids=pt_inputs['input_ids'],
        attention_mask=pt_inputs['attention_mask'],
        max_new_tokens=8
    )
baseline_preds = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)
true_labels = ["label_0","label_1","label_2","label_3"]
true_labels = [true_labels[t] for t in pt_targets['input_ids'].squeeze(1).tolist()]
baseline_em = em_score(baseline_preds, true_labels)

for ex in tqdm(test_errors, desc="Refinement loop"):
    # Fine‑tune on the error
    model = fine_tune_single(ex, steps=5, lr=5e-5)
    # Predict forgotten examples
    with torch.no_grad():
        # encode online example
        inputs, _ = encode_examples([ex])
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        online_repr = rep_forecaster.encode(model.base_model.encoder(**inputs).last_hidden_state)
        # compute scores for all pt examples
        scores = []
        for j in range(NUM_PT_EXAMPLES):
            pt_input = {
                'input_ids': pt_inputs['input_ids'][j:j+1],
                'attention_mask': pt_inputs['attention_mask'][j:j+1]
            }
            pt_repr = rep_forecaster.encode(model.base_model.encoder(**pt_input).last_hidden_state)
            logits = rep_forecaster(online_repr, pt_repr, biases[j].unsqueeze(0))
            scores.append(torch.sigmoid(logits).item())
        scores = np.array(scores)
        topk_idx = scores.argsort()[-k_replay:][::-1]
        # Replay those examples (fine‑tune on them again)
        for idx in topk_idx:
            # prepare example
            pt_example = {
                'text': pt_examples[idx]['text'],
                'label': pt_examples[idx]['label']
            }
            model = fine_tune_single(pt_example, steps=2, lr=5e-5)

    # Evaluate edit success on the current error
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=8
        )
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    true = ["label_0","label_1","label_2","label_3"][ex['label']]
    edit_success.append(int(pred.strip() == true.strip()))

    # EM after this error
    with torch.no_grad():
        outputs = model.generate(
            input_ids=pt_inputs['input_ids'],
            attention_mask=pt_inputs['attention_mask'],
            max_new_tokens=8
        )
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    em = em_score(preds, true_labels)
    em_drop_list.append(baseline_em - em)

# Aggregate results
edit_success_rate = np.mean(edit_success)
em_drop_ratio = np.mean(em_drop_list)

print("[+] Replay refinement results:")
print(f"Edit Success Rate: {edit_success_rate:.3f}")
print(f"EM Drop Ratio: {em_drop_ratio:.3f}%")

replay_metrics = {
    "edit_success_rate": edit_success_rate,
    "em_drop_ratio": em_drop_ratio
}
with open("replay_results.pkl", "wb") as f:
    pickle.dump(replay_metrics, f)