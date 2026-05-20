import os
import torch
import yaml
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from utils import set_seed, ensure_dir

config = yaml.safe_load(open("config.yaml"))
set_seed(config["seed"])
device = config["device"]
dpo_cfg = config["dpo"]

tokenizer = AutoTokenizer.from_pretrained(dpo_cfg["base_model"])
base_model = AutoModelForCausalLM.from_pretrained(dpo_cfg["base_model"]).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(dpo_cfg["base_model"]).to(device)

# Load pairwise data
pairwise_ds = load_from_disk("outputs/pairwise_dataset")

trainer = DPOTrainer(
    model=base_model,
    ref_model=ref_model,
    train_dataset=pairwise_ds,
    eval_dataset=None,
    tokenizer=tokenizer,
    config=DPOConfig(
        beta=dpo_cfg["beta"],
        learning_rate=dpo_cfg["lr"],
        max_new_tokens=dpo_cfg["max_new_tokens"],
    ),
    args={
        "per_device_train_batch_size": dpo_cfg["batch_size"],
        "gradient_accumulation_steps": 1,
        "max_steps": -1,
        "num_train_epochs": dpo_cfg["epochs"],
        "logging_steps": 10,
        "save_strategy": "epoch",
        "output_dir": dpo_cfg["output_path"],
    },
)

trainer.train()
trainer.save_model(dpo_cfg["output_path"])
print("DPO training finished. Model saved to", dpo_cfg["output_path"])