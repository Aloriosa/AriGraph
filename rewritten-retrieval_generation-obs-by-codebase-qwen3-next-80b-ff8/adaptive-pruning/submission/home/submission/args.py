import argparse
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: "},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    t_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the training and validation files."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            self.t_name = self.t_name.lower()
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    apply_lora: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA adapters to the model"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )

@dataclass
class MinusTrainingArguments:
    """
    Arguments pertaining to what training strategies we are going to apply.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_dir: Optional[str] = field(
        default=None, metadata={"help": "Tensorboard log dir."}
    )
    logging_first_step: bool = field(
        default=False, metadata={"help": "Log and eval the first global_step"}
    )
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Deprecated, the use of `--debug` flag is deprecated. "
                "Whether to print debug metrics on TPU."
            )
        },
    )
    debug: str = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded "
                "in the main process."
            )
        },
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[list] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same "
                "training data."
            )
        },
    )
    sharded_ddp: str = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use sharded DDP training (in distributed training only). The base option "
                "is `simple`, `zero_dp_2`, `zero_dp_3`."
            )
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    optim: str = field(default="adamw_hf", metadata={"help": "The optimizer to use."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    fp16_backend: str = field(
        default="auto",
        metadata={"help": "The backend to be used for mixed precision."},
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )
    # APT specific arguments
    pruner_type: str = field(
        default="running_fisher",
        metadata={"help": "Type of pruner to use (running_fisher, magnitude, etc.)"}
    )
    param_allocation_strategy: str = field(
        default="running_fisher",
        metadata={"help": "Strategy for parameter allocation (running_fisher, random, etc.)"}
    )
    mac_constraint: float = field(
        default=0.7,
        metadata={"help": "Target MAC constraint (sparsity level) for pruning"}
    )
    pre_tuning_constraint: float = field(
        default=1.0,
        metadata={"help": "Pre-tuning constraint for initial pruning"}
    )
    pruning_start: int = field(
        default=-1,
        metadata={"help": "Step to start pruning (-1 means start immediately)"}
    )
    pruning_stop: int = field(
        default=5,
        metadata={"help": "Step to stop pruning"}
    )
    num_prunings: int = field(
        default=16,
        metadata={"help": "Number of pruning steps"}
    )
    pruning_batches: int = field(
        default=128,
        metadata={"help": "Number of batches to use for salience calculation"}
    )
    pruning_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for pruning salience calculation"}
    )
    pre_pruning_tuning_steps: int = field(
        default=200,
        metadata={"help": "Number of steps for pre-pruning tuning"}
    )
    sparsity_warmup_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for sparsity warmup"}
    )
    distillation_type: str = field(
        default="self_momentum",
        metadata={"help": "Type of distillation (self_momentum, teacher_student, etc.)"}
    )
    distill_mapping_strategy: str = field(
        default="dynamic_block_teacher_dynamic_student",
        metadata={"help": "Strategy for distillation mapping"}
    )
    teacher_param_tuning_config: str = field(
        default="dq:0-31,dv:0-31",
        metadata={"help": "Teacher parameter tuning configuration"}
    )
    student_param_tuning_config: str = field(
        default="dq:0-31,dv:0-31",
        metadata={"help": "Student parameter tuning configuration"}
    )
    warmup_param_tuning_config: str = field(
        default="dq:0-31,dv:0-31",
        metadata={"help": "Warmup parameter tuning configuration"}
    )
    collect_salience: bool = field(
        default=True,
        metadata={"help": "Whether to collect salience scores during training"}
    )
    continuous_allocation: bool = field(
        default=False,
        metadata={"help": "Whether to use continuous parameter allocation"}
    )
    continuous_alloc_interval: int = field(
        default=100,
        metadata={"help": "Interval for continuous parameter allocation"}
    )
    restore_before_pruning: bool = field(
        default=False,
        metadata={"help": "Whether to restore model before pruning"}
    )
    pre_tuning_scorer: str = field(
        default="kurtosis",
        metadata={"help": "Pre-tuning scorer type"}
    )
    pre_tuning_pruner: str = field(
        default="magnitude",
        metadata={"help": "Pre-tuning pruner type"}
    )
    adapter_type: str = field(
        default="lora",
        metadata={"help": "Type of adapter to use (lora, adapter, etc.)"}
    )
    max_lora_r: int = field(
        default=8,
        metadata={"help": "Maximum LoRA rank"}
    )