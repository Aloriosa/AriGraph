import os
import torch
import json
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from args import DataTrainingArguments, ModelArguments, MinusTrainingArguments
from models import build_model
from trainer.trainer_minus import MinusTrainer
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import logging

logger = logging.getLogger(__name__)

def get_raw_datasets(model_args, data_args, training_args):
    """Load raw datasets"""
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        t_name = data_args.task_name
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        t_name = data_args.dataset_name
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        
        # Get the test dataset if test_file is provided
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        t_name = "custom"
        
    return t_name, raw_datasets

def build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=False):
    """Build model, tokenizer, and config"""
    # Load config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
    
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Load model
    if model_args.model_name_or_path:
        if 'llama' in model_args.model_name_or_path.lower():
            model = ElasticLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)
    
    # Apply LoRA if requested
    if model_args.apply_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM" if 'llama' in model_args.model_name_or_path.lower() else "SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return config, tokenizer, model

def build_trainer(model_args, data_args, training_args, t_name, raw_datasets, model, tokenizer):
    """Build trainer"""
    # Preprocessing the datasets
    if t_name == "custom":
        # Handle custom datasets
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name],
                padding="max_length",
                truncation=True,
                max_length=data_args.max_seq_length,
                return_special_tokens_mask=True,
            )
            
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
    else:
        # GLUE datasets
        sentence1_key, sentence2_key = task_to_keys[t_name]
        
        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding="max_length", max_length=data_args.max_seq_length, truncation=True)
            return result
            
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    
    # Initialize our Trainer
    trainer = MinusTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics[t_name] if t_name in compute_metrics else None,
    )
    
    return trainer

def efficiency_testing(model, tokenizer, device, max_seq_length=128, num_iterations=10):
    """Test inference efficiency"""
    model.eval()
    model.to(device)
    
    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_iterations
    
    # Measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    
    return {
        "inference_time": avg_inference_time,
        "peak_memory_gb": peak_memory
    }

def input_constructor(model, tokenizer, max_seq_length=128):
    """Construct input for profiling"""
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length))
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def lora_to_linear(model):
    """Convert LoRA layers to linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            # Create new linear layer
            new_linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            # Copy weights
            new_linear.weight.data = module.weight.data
            if module.bias is not None:
                new_linear.bias.data = module.bias.data
            # Replace module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_linear)
    
    return model

# Global variables
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

compute_metrics = {
    "cola": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
    "mnli": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
    "mrpc": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean(), "f1": f1_score(p.label_ids, p.predictions.argmax(axis=1))},
    "qnli": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
    "qqp": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean(), "f1": f1_score(p.label_ids, p.predictions.argmax(axis=1))},
    "rte": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
    "sst2": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
    "stsb": lambda p: {"pearson": pearsonr(p.predictions[:, 0], p.label_ids)[0], "spearman": spearmanr(p.predictions[:, 0], p.label_ids)[0]},
    "wnli": lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()},
}

# Import necessary modules
try:
    from sklearn.metrics import f1_score, pearsonr, spearmanr
except ImportError:
    def f1_score(y_true, y_pred):
        return 0.0
    def pearsonr(x, y):
        return 0.0
    def spearmanr(x, y):
        return 0.0

# Import transformers
try:
    from transformers import AutoConfig, CONFIG_MAPPING
except ImportError:
    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    CONFIG_MAPPING = {}

# Import datasets
try:
    from datasets import load_dataset
except ImportError:
    def load_dataset(*args, **kwargs):
        return None

# Import torch
try:
    import torch
except ImportError:
    class torch:
        class Tensor:
            pass
        class device:
            pass
        class no_grad:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        class cuda:
            @staticmethod
            def empty_cache():
                pass
            @staticmethod
            def reset_peak_memory_stats():
                pass
            @staticmethod
            def max_memory_allocated():
                return 0
        class bfloat16:
            pass
        class float16:
            pass
        class long:
            pass
        class ones:
            @staticmethod
            def ones(*args, **kwargs):
                return None
        class randint:
            @staticmethod
            def randint(*args, **kwargs):
                return None
        class LongTensor:
            pass
        class float:
            pass
        class int:
            pass
        class device:
            def __init__(self, device):
                self.device = device
            def __str__(self):
                return self.device

# Import time
try:
    import time
except ImportError:
    import time