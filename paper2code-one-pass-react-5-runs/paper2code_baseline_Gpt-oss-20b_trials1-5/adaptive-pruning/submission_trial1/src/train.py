import os
import yaml
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score
from src.apt_adapter import APTAdapter
from src.utils import prune_attention_heads, adjust_adapter_rank

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Set deterministic
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
dataset = load_dataset("glue", "sst2")

# Preprocess
def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=cfg["max_seq_length"])

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"])
model.config.num_labels = 2  # SST-2
model.config.problem_type = "single_label_classification"

# Insert APT adapters into each transformer layer (encoder only)
for i, layer in enumerate(model.base_model.encoder.layer):
    adapter = APTAdapter(hidden_size=layer.attention.self.query.weight.shape[0],
                         rank=cfg["initial_rank"],
                         scale=1.0)
    layer.attention.self.query = nn.Sequential(
        layer.attention.self.query,
        adapter
    )
    # Store adapter reference for dynamic rank adjustment
    layer.apt_adapter = adapter

# Prune attention heads early
prune_attention_heads(model, cfg["head_prune_percent"])

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    per_device_eval_batch_size=cfg["batch_size"],
    learning_rate=cfg["learning_rate"],
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    fp16=True,
    report_to="none",
)

# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("=== APT Training ===")
trainer.train()

# Evaluate on test set
print("\n=== APT Evaluation ===")
test_dataset = dataset["test"]
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
test_acc = accuracy_score(test_dataset["label"], preds)
print(f"Test accuracy: {test_acc:.4f}")

# Save accuracy to file for the grading script
os.makedirs("output", exist_ok=True)
with open("output/accuracy.txt", "w") as f:
    f.write(f"{test_acc:.4f}\n")

# Report model size after pruning
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
size_mb = total_params * 4 / (1024 ** 2)  # float32
print(f"\nModel size after pruning: {size_mb:.2f} MB")