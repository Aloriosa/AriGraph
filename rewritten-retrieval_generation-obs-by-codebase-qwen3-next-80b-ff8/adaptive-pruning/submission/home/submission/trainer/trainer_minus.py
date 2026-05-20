import torch
import torch.nn as nn
import logging
import time
import math
import os
from typing import Optional, List, Dict, Any
from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, speed_metrics
from transformers.data.data_collator import DataCollator
from datasets import Dataset
from torch.utils.data import Subset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers import HfArgumentParser
from args import DataTrainingArguments, ModelArguments, MinusTrainingArguments
from models import build_model
from prune.pruner import AdapterPruner
from trainer.param_control import ParamController
from transformers import Trainer
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MinusTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # APT-specific arguments
        self.pruner_type = kwargs.get('pruner_type', 'running_fisher')
        self.param_allocation_strategy = kwargs.get('param_allocation_strategy', 'running_fisher')
        self.mac_constraint = kwargs.get('mac_constraint', 0.7)
        self.pre_tuning_constraint = kwargs.get('pre_tuning_constraint', 1.0)
        self.pruning_start = kwargs.get('pruning_start', -1)
        self.pruning_stop = kwargs.get('pruning_stop', 5)
        self.num_prunings = kwargs.get('num_prunings', 16)
        self.pruning_batches = kwargs.get('pruning_batches', 128)
        self.pruning_batch_size = kwargs.get('pruning_batch_size', 2)
        self.pre_pruning_tuning_steps = kwargs.get('pre_pruning_tuning_steps', 200)
        self.sparsity_warmup_epochs = kwargs.get('sparsity_warmup_epochs', 1)
        self.distillation_type = kwargs.get('distillation_type', 'self_momentum')
        self.distill_mapping_strategy = kwargs.get('distill_mapping_strategy', 'dynamic_block_teacher_dynamic_student')
        self.teacher_param_tuning_config = kwargs.get('teacher_param_tuning_config', 'dq:0-31,dv:0-31')
        self.student_param_tuning_config = kwargs.get('student_param_tuning_config', 'dq:0-31,dv:0-31')
        self.warmup_param_tuning_config = kwargs.get('warmup_param_tuning_config', 'dq:0-31,dv:0-31')
        self.collect_salience = kwargs.get('collect_salience', True)
        self.continuous_allocation = kwargs.get('continuous_allocation', False)
        self.continuous_alloc_interval = kwargs.get('continuous_alloc_interval', 100)
        self.restore_before_pruning = kwargs.get('restore_before_pruning', False)
        self.pre_tuning_scorer = kwargs.get('pre_tuning_scorer', 'kurtosis')
        self.pre_tuning_pruner = kwargs.get('pre_tuning_pruner', 'magnitude')
        self.adapter_type = kwargs.get('adapter_type', 'lora')
        self.max_lora_r = kwargs.get('max_lora_r', 8)

class MinusTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: MinusTrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            eval_examples = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            param_controller: Optional[ParamController] = None,
            teacher_model: Optional[PreTrainedModel] = None,
            seq_len: int = 128,
            output_seq_len: Optional[int] = None,
            cls_task: bool = True,
            pre_tune_head_mask: Optional[torch.Tensor] = None,
            pre_tune_intermediate_mask: Optional[torch.Tensor] = None,
            post_processing_function=None,
    ):

        print("Model is None: ", model is None)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Setup logging
        fileHandler = logging.FileHandler("{0}/{1}.log".format(args.output_dir, 'trainer'))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        
        # Parse tuning configurations
        teacher_config = ParamController.parse_tuning_param_str(args.teacher_param_tuning_config)
        logger.info("Teacher config: " + str(teacher_config))
        student_config = ParamController.parse_tuning_param_str(args.student_param_tuning_config)
        logger.info("Student config: " + str(student_config))
        
        # Model architecture setup
        self.model_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.num_layers
        self.model_encoder_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.num_layers
        self.model_decoder_layers = model.config.num_decoder_layers if hasattr(model.config, 'num_decoder_layers') else None
        if model.base_model_prefix == 'transformer':
            self.attention_head_size = model.config.d_kv
            self.t5_backbone = True
        elif 'bert' in model.base_model_prefix or model.config.model_type == 'llama':
            self.attention_head_size = model.config.hidden_size // model.config.num_attention_heads
            self.t5_backbone = False
        self.gated_ffn = 'gated' in getattr(model.config, 'feed_forward_proj', '') or model.config.model_type == 'llama'
        
        # Setup warmup config
        if args.warmup_param_tuning_config is None:
            warmup_config = {
                k: list(range(self.model_layers)) for k in teacher_config
            } if teacher_config is not None else None
        else:
            warmup_config = ParamController.parse_tuning_param_str(args.warmup_param_tuning_config)
        logger.info("Warmup config: " + str(warmup_config))

        # Setup pruning dataloader
        self.pruning_dataloader = DataLoader(
            Subset(train_dataset, torch.randperm(len(train_dataset)).tolist()[:args.pruning_batch_size * args.pruning_batches]),
            batch_size = args.pruning_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        ) if train_dataset is not None else None
        
        # Setup parameter controller
        if args.param_allocation_strategy == 'none':
            self.param_dynamic_allocation = False
        else:
            self.param_dynamic_allocation = True
        adapter_pruner = AdapterPruner(self.model, self.pruning_dataloader)
        logger.info("Adapter type: " + args.adapter_type)
        
        if param_controller is not None:
            self.param_controller = param_controller
        else:
            self.param_controller = ParamController(
                self.model,
                args=args,
                teacher_config=teacher_config,
                student_config=student_config,
                warmup_config=warmup_config,
                lora_with_bias=False,
                adapter_pruner=adapter_pruner,
                param_allocation_strategy=args.param_allocation_strategy,
                adapter_type=args.adapter_type,
                max_lora_r=args.max_lora_r,
            )
        
        # Post-processing function for SQuAD v2
        self.post_processing_function = post_processing_function
        self.eval_examples = eval_examples
        self.best_eval = None
        self.pruning_conducted = False
        self.distilling = False
        self.distill_finished = False
        self.distill_step_cnt = 0
        self.moving_term = None
        self.now_distill_loss = 0
        self.now_distill_ce_loss = 0
        self.teacher_tuning_params = None
        self.distill_tuning_params = []
        self.pruning_tuning_params = []
        self.current_model_ratio = 1
        self.final_tuning_params = None
        self.distillation_is_self = teacher_model is None and 'momentum' not in self.args.distillation_type
        self.teacher_model = model if self.distillation_is_self else teacher_model
        self.teacher_model_masks = None
        self.pruning_start_step = None
        self.pruning_end_step = None
        self.pruning_steps = None
        self.pre_pruning_tuning_steps = None
        self.last_pruning_step = 0
        self.mask_decaying = False
        self.num_prunings = None
        self.distill_start_step = -1
        self.distill_steps = -1
        self.layer_warmup_steps = -1
        self.teacher_distillation_learning = args.teacher_learning
        self.epoch_start_mem, self.epoch_end_mem = None, None
        self.mac_constraint_schedule = None
        self.current_mac_constraint = None
        self.current_distillation_type = None
        self.distill_mapping_strategy = self.args.distill_mapping_strategy
        self.auto_layer_conversion = True
        self.salience_to_be_collected = self.args.collect_salience
        self.collecting_salience = False
        self.save_salience = False
        self.salience_collecting_start = self.args.salience_collecting_start
        self.salience_collecting_end = self.args.salience_collecting_end
        self.salience_collected = []
        
        # Setup pruning scheduler
        if self.args.pre_tuning_scorer != 'none' and self.args.pre_tuning_constraint < 1:
            self.pre_tuning_pruning_scorer = build_scorer(self.args.pre_tuning_scorer, self.model, self.pruning_dataloader, param_controller=self.param_controller, state=self.state, gather_freq=1, beta_1=self.beta_1, beta_2=self.beta_2, use_uncertainty=False, block_normalize_dict=self.block_normalize_term, static=True, use_kurtosis=True)
            self.pre_tuning_pruning_pruner = build_pruner(args.pre_tuning_pruner, args, model, self.pre_tuning_pruning_scorer)
            self.starting_mac_constraint = self.args.pre_tuning_constraint
        else:
            self.pre_tuning_pruning_scorer = None
            self.pre_tuning_pruning_pruner = None
            self.starting_mac_constraint = 1
            
        if self.args.pruner_type != 'none':
            self.scorer = build_scorer('backward_running_hidden_states_salience', self.model, None, param_controller = self.param_controller, state=self.state, gather_freq=1, beta_1=self.beta_1, beta_2=self.beta_2, use_uncertainty=False, block_normalize_dict=self.block_normalize_term, static=False, use_kurtosis=True)
        else:
            self.scorer = None
        self.pruner = build_pruner(args.pruner_type,args, model, self.scorer)
        
        # Setup evaluation key
        task_name = getattr(args, 'task_name', None)
        self.eval_key = task2evalkey[task_name] if task_name in task2evalkey else 'eval_loss'
        self.bigger_is_better = self.eval_key != 'eval_loss'
        self.best_teacher_metric = 0. if self.bigger_is_better else 1e10
        
        # Setup mixed precision
        if (args.fp16 or args.bf16) and self.sharded_ddp is None:
            if args.half_precision_backend == "auto":
                if args.device == torch.device("cpu"):
                    if args.fp16:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                    else:
                        args.half_precision_backend = "cpu_amp"
                else:
                    args.half_precision_backend = "cuda_amp"

            logger.info(f"Using {args.half_precision_backend} half precision backend")

        self.do_grad_scaling = False
        if (args.fp16 or args.bf16):
            if self.sharded_ddp is None:
                if args.half_precision_backend == "cuda_amp":
                    self.use_cuda_amp = True
                    self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
                elif args.half_precision_backend == "cpu_amp":
                    self.use_cpu_amp = True
                    self.amp_dtype = torch.bfloat16
                    
        logger.info("Half precision backend: " + args.half_precision_backend)
        logger.info("Half precision dtype: " + str(getattr(self, 'amp_dtype', None)))
        
        # Setup pruning schedule
        self._setup_pruning_schedule()
        
    def _setup_pruning_schedule(self):
        """Setup pruning schedule based on paper"""
        total_steps = self.args.num_train_epochs * len(self.train_dataloader)
        
        # Calculate pruning steps
        if self.args.pruning_start == -1:
            self.pruning_start_step = 0
        else:
            self.pruning_start_step = self.args.pruning_start
            
        if self.args.pruning_stop == -1:
            self.pruning_end_step = total_steps
        else:
            self.pruning_end_step = self.args.pruning_stop
            
        self.pruning_steps = self.pruning_end_step - self.pruning_start_step
        self.num_prunings = self.args.num_prunings
        
        # Create MAC constraint schedule
        self.mac_constraint_schedule = []
        start_constraint = self.starting_mac_constraint
        end_constraint = self.args.mac_constraint
        
        # Create linear schedule for pruning
        for i in range(self.num_prunings + 1):
            step = self.pruning_start_step + (self.pruning_end_step - self.pruning_start_step) * i / self.num_prunings
            constraint = start_constraint - (start_constraint - end_constraint) * i / self.num_prunings
            self.mac_constraint_schedule.append((int(step), constraint))
        
        # Add final constraint
        self.mac_constraint_schedule.append((total_steps, end_constraint))
        
        # Setup pre-pruning tuning
        self.pre_pruning_tuning_steps = self.args.pre_pruning_tuning_steps
        
        logger.info(f"Pruning schedule: {self.mac_constraint_schedule}")
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, return_dict=False)
        
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Apply adaptive pruning and tuning at appropriate steps
        self._apply_adaptive_pruning_and_tuning()
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
            
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
            
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _apply_adaptive_pruning_and_tuning(self):
        """Apply adaptive pruning and tuning based on current step"""
        current_step = self.state.global_step
        
        # Apply pre-pruning tuning
        if current_step < self.pre_pruning_tuning_steps:
            # Apply initial tuning without pruning
            self.param_controller.apply_adaptive_pruning(current_step, 1.0)
            return
            
        # Apply pruning schedule
        for step, constraint in self.mac_constraint_schedule:
            if current_step >= step:
                self.current_mac_constraint = constraint
            else:
                break
                
        # Apply pruning if within pruning window
        if self.pruning_start_step <= current_step <= self.pruning_end_step:
            # Calculate which pruning step we're on
            pruning_step = (current_step - self.pruning_start_step) // (self.pruning_steps // self.num_prunings)
            
            # Apply pruning
            if pruning_step > 0 and (current_step - self.last_pruning_step) >= self.pruning_steps // self.num_prunings:
                # Generate new mask based on salience
                mask = self.pruner.generate_mask(self.current_mac_constraint, current_step)
                self.param_controller.apply_pruning_mask(mask)
                self.last_pruning_step = current_step
                
                logger.info(f"Applied pruning at step {current_step} with constraint {self.current_mac_constraint}")
        
        # Apply adaptive tuning
        tuning_params, current_rank = self.param_controller.get_tuning_parameters(current_step)
        self.param_controller.apply_adaptive_tuning(tuning_params, current_rank)
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_processing_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_processing_function(
                self.eval_examples,
                eval_dataset,
                output.predictions
            )
            metrics = self.compute_metrics(eval_preds)
        else:
            metrics = {}
        
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        metrics.update(output.metrics)
        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
    
    def _get_distillation_loss(self, student_outputs, teacher_outputs):
        """Calculate distillation loss"""
        if self.args.distillation_type == 'self_momentum':
            # Self-distillation with momentum
            return self._self_distillation_loss(student_outputs, teacher_outputs)
        else:
            # Standard distillation
            return self._standard_distillation_loss(student_outputs, teacher_outputs)
    
    def _self_distillation_loss(self, student_outputs, teacher_outputs):
        """Self-distillation with momentum"""
        # Use student outputs as both student and teacher
        # This is a simplified implementation
        student_logits = student_outputs.logits
        teacher_logits = student_outputs.logits  # Use same as student
        
        # KL divergence loss
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        loss = loss_fct(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1)
        ) * (self.args.temperature ** 2)
        
        return loss
    
    def _standard_distillation_loss(self, student_outputs, teacher_outputs):
        """Standard distillation loss"""
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # KL divergence loss
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        loss = loss_fct(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1)
        ) * (self.args.temperature ** 2)
        
        return loss