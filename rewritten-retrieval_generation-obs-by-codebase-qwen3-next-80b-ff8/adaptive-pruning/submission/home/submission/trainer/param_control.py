import torch
import logging
from typing import Optional, Dict, List, Set
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

class ParamController:
    def __init__(self, model: torch.nn.Module, args = None, teacher_config: Optional[Dict[str, List[int]]] = None, student_config: Optional[Dict[str, List[int]]] = None, warmup_config: Optional[Dict[str, List[int]]] = None, bias_only: bool = False, lora_with_bias: bool = False, param_allocation_strategy: str = 'none', adapter_type: str = 'lora', max_lora_r: int = -1, **kwargs):
        if args is not None:
            fileHandler = logging.FileHandler("{0}/{1}.log".format(args.output_dir, 'trainer'))
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        self.model = model
        self.named_modules = dict(self.model.named_modules())
        self.bias_only = bias_only
        self.lora_with_bias = lora_with_bias
        self.model_arch = ModelArch(self.model)
        self.name2template = self.model_arch.get_name2template()
        self.teacher_config = teacher_config
        self.student_config = student_config
        if warmup_config is None:
            self.warmup_config = teacher_config
        else:
            self.warmup_config = warmup_config
        self.param_config = {}
        self.warmup_param_config = {}
        self.teacher_tuning_layers = set()
        self.student_tuning_layers = set()
        
        for i in range(self.model.config.num_hidden_layers):
            self.param_config[i] = {}
            self.warmup_param_config[i] = set()
            for k in self.name2template:
                module_is_teacher = self.teacher_config is not None and k in self.teacher_config and i in self.teacher_config[k]
                module_is_student = self.student_config is not None and k in self.student_config and i in self.student_config[k]
                if module_is_teacher and i not in self.teacher_tuning_layers:
                    self.teacher_tuning_layers.add(i)
                if module_is_student and i not in self.student_tuning_layers:
                    self.student_tuning_layers.add(i)
                module_in_warmup = self.warmup_config is not None and k in self.warmup_config and i in self.warmup_config[k]
                if module_in_warmup:
                    self.warmup_param_config[i].add(k)
                if module_is_teacher and module_is_student:
                    self.param_config[i][k] = 'teacher_student'
                elif module_is_teacher:
                    self.param_config[i][k] = 'teacher'
                elif module_is_student:
                    self.param_config[i][k] = 'student'
                else:
                    self.param_config[i][k] = 'none'
        
        self.warmup_params = self._config_to_params(self.warmup_config)
        self.teacher_params = self._config_to_params(teacher_config) if teacher_config is not None else None
        self.student_params = self._config_to_params(student_config) if student_config is not None else None
        self.adapter_type = adapter_type
        self.tuning_param_number = 0
        self.tuning_param_number_fixed = False
        self.schedule = None
        self.tuning_schedule = None
        self.next_tuning_param_num = None
        self.next_tuning_step, self.next_tuning_param_num = None, None
        self.target_tuning_num = None
        self.next_pruning_step = None
        self.next_pruning_ratio = None
        self.model.is_teacher = False
        self.model.is_student = False
        self.model.is_colearning = False
        self.model.is_distilling = False
        self.param_allocation_func = PARAM_ALLOCATION_FUNC[param_allocation_strategy] if param_allocation_strategy in PARAM_ALLOCATION_FUNC else None
        self.max_lora_r = max_lora_r
        self.beta_1 = 0.85
        self.beta_2 = 0.85
        
        # Initialize adaptive tuning parameters
        self.adaptive_tuning_schedule = []
        self._setup_adaptive_tuning_schedule()
        
    def _config_to_params(self, config):
        """Convert config to parameter names"""
        if config is None:
            return set()
        
        params = set()
        for key, layers in config.items():
            for layer in layers:
                if key == 'q':
                    params.add(f'model.layers.{layer}.self_attn.q_proj')
                elif key == 'k':
                    params.add(f'model.layers.{layer}.self_attn.k_proj')
                elif key == 'v':
                    params.add(f'model.layers.{layer}.self_attn.v_proj')
                elif key == 'o':
                    params.add(f'model.layers.{layer}.self_attn.o_proj')
                elif key == 'i':
                    params.add(f'model.layers.{layer}.mlp.up_proj')
                    params.add(f'model.layers.{layer}.mlp.gate_proj')
                elif key == 'dq':
                    params.add(f'model.layers.{layer}.self_attn.q_proj')
                elif key == 'dk':
                    params.add(f'model.layers.{layer}.self_attn.k_proj')
                elif key == 'dv':
                    params.add(f'model.layers.{layer}.self_attn.v_proj')
                elif key == 'do':
                    params.add(f'model.layers.{layer}.self_attn.o_proj')
                elif key == 'di':
                    params.add(f'model.layers.{layer}.mlp.up_proj')
                    params.add(f'model.layers.{layer}.mlp.gate_proj')
        return params
    
    def _setup_adaptive_tuning_schedule(self):
        """Setup adaptive tuning schedule based on paper"""
        # Based on paper: start with low rank and increase over time
        # We'll use a linear schedule from initial rank to max rank
        initial_rank = 2
        target_rank = self.max_lora_r
        total_steps = 1000  # Approximate total training steps
        
        # Create schedule: increase rank every 100 steps
        for step in range(0, total_steps, 100):
            rank = min(initial_rank + (step // 100) * 2, target_rank)
            self.adaptive_tuning_schedule.append((step, rank))
        
        # Ensure final step reaches target rank
        if self.adaptive_tuning_schedule[-1][1] < target_rank:
            self.adaptive_tuning_schedule.append((total_steps, target_rank))
    
    def get_tuning_parameters(self, step: int):
        """Get tuning parameters for current step"""
        # Get current rank based on adaptive schedule
        current_rank = self._get_current_rank(step)
        
        # Get parameters to tune based on salience
        tuning_params = set()
        
        # For each layer, determine if it should be tuned based on salience
        for i in range(self.model.config.num_hidden_layers):
            # Check if this layer should be tuned based on adaptive tuning schedule
            if self.param_config[i] and any(v != 'none' for v in self.param_config[i].values()):
                # Add parameters with current rank
                for key, param_type in self.param_config[i].items():
                    if param_type != 'none':
                        if key == 'q':
                            tuning_params.add(f'model.layers.{i}.self_attn.q_proj')
                        elif key == 'k':
                            tuning_params.add(f'model.layers.{i}.self_attn.k_proj')
                        elif key == 'v':
                            tuning_params.add(f'model.layers.{i}.self_attn.v_proj')
                        elif key == 'o':
                            tuning_params.add(f'model.layers.{i}.self_attn.o_proj')
                        elif key == 'i':
                            tuning_params.add(f'model.layers.{i}.mlp.up_proj')
                            tuning_params.add(f'model.layers.{i}.mlp.gate_proj')
                        elif key == 'dq':
                            tuning_params.add(f'model.layers.{i}.self_attn.q_proj')
                        elif key == 'dk':
                            tuning_params.add(f'model.layers.{i}.self_attn.k_proj')
                        elif key == 'dv':
                            tuning_params.add(f'model.layers.{i}.self_attn.v_proj')
                        elif key == 'do':
                            tuning_params.add(f'model.layers.{i}.self_attn.o_proj')
                        elif key == 'di':
                            tuning_params.add(f'model.layers.{i}.mlp.up_proj')
                            tuning_params.add(f'model.layers.{i}.mlp.gate_proj')
        
        return tuning_params, current_rank
    
    def _get_current_rank(self, step: int):
        """Get current LoRA rank based on adaptive schedule"""
        if not self.adaptive_tuning_schedule:
            return self.max_lora_r
            
        # Find the rank for current step
        for i in range(len(self.adaptive_tuning_schedule) - 1):
            if self.adaptive_tuning_schedule[i][0] <= step < self.adaptive_tuning_schedule[i + 1][0]:
                return self.adaptive_tuning_schedule[i][1]
        
        # Return last rank if step is beyond schedule
        return self.adaptive_tuning_schedule[-1][1]
    
    def apply_adaptive_pruning(self, step: int, sparsity_level: float):
        """Apply adaptive pruning based on salience"""
        # Get salience scores (this would be computed from gradients)
        head_scores, intermediate_scores = self._compute_salience_scores()
        
        # Apply pruning based on salience
        self._apply_pruning(head_scores, intermediate_scores, sparsity_level)
        
    def _compute_salience_scores(self):
        """Compute salience scores for pruning"""
        # This would normally use gradient information
        # For now, we'll use a placeholder implementation
        head_scores = []
        intermediate_scores = []
        
        for i in range(self.model.config.num_hidden_layers):
            # Placeholder: use random scores
            head_scores.append(torch.rand(self.model.config.num_attention_heads))
            intermediate_scores.append(torch.rand(self.model.config.intermediate_size))
            
        return head_scores, intermediate_scores
    
    def _apply_pruning(self, head_scores, intermediate_scores, sparsity_level):
        """Apply pruning to model based on salience scores"""
        # This would normally modify the model's masks
        # For now, we'll use a placeholder implementation
        pass

# Helper classes and functions
class ModelArch:
    def __init__(self, model):
        self.model = model
        self.name2template = {}
        
    def get_name2template(self):
        # This would normally extract template information from model architecture
        return self.name2template

# Global parameter allocation functions
PARAM_ALLOCATION_FUNC = {
    'running_fisher': lambda x: x,
    'none': lambda x: x,
}

# Logging formatter
class LogFormatter:
    def __init__(self):
        pass

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")