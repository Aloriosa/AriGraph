import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
import math
from typing import Optional, Dict, List, Tuple

class ElasticLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).view(-1)
        self.intermediate_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).view(-1)
        self.hidden_mask = torch.ones(config.hidden_size)
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.retained_indices = None
        self.virtual_pruned = False
        self.is_teacher = False
        self.is_student = True
        self.is_distilling = False
        self.is_colearning = False
        
        if hasattr(config, 'do_distill') and config.do_distill:
            if config.apply_lora:
                self.layer_transformation = lora.PruningLinear(config.hidden_size, config.hidden_size, r=8, lora_alpha=16, bias=False, dtype=self.dtype)
            else:
                self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=self.dtype)
        else:
            self.layer_transformation = None
        
        self.head_layer_z = torch.ones(config.num_hidden_layers)
        self.mlp_z = torch.ones(config.num_hidden_layers)
            
    def clear_masks(self):
        self.head_mask = None
        self.intermediate_mask = None
        self.hidden_mask = None
        
    def reset_masks(self):
        head_nums = [self.model.layers[i].self_attn.num_heads for i in range(self.config.num_hidden_layers)]
        intermediate_sizes = [self.model.layers[i].mlp.gate_proj.weight.shape[0] for i in range(self.config.num_hidden_layers)]
        self.head_mask = torch.ones(sum(head_nums)).view(-1).to(self.device)
        self.intermediate_mask = torch.ones(sum(intermediate_sizes)).view(-1).to(self.device)
        
    def _mask_fine_to_coarse(self, mask):
        if isinstance(mask, torch.Tensor):
            return mask.detach().any(dim=-1).float()
        elif isinstance(mask, list) or isinstance(mask, tuple):
            return [v.detach().any().float().unsqueeze(0) if v is not None else None for v in mask]
        
    def split_mask_or_score(self, head_mask = None, intermediate_mask = None):
        head_mask = self.head_mask if head_mask is None else head_mask
        intermediate_mask = self.intermediate_mask if intermediate_mask is None else intermediate_mask
        return split_mask_or_score(self, head_mask, intermediate_mask)
    
    def update_layer_z(self):
        head_mask, intermediate_mask = self.split_mask_or_score(self.backup_head_mask, self.backup_intermediate_mask) if self.virtual_pruned else self.split_mask_or_score(self.head_mask, self.intermediate_mask)
        self.head_layer_z = torch.cat(self._mask_fine_to_coarse(head_mask)).float()
        self.mlp_z = torch.cat(self._mask_fine_to_coarse(intermediate_mask)).float()
        
    def prune_model_with_masks(self, continual_pruning=True):
        assert isinstance(self.config, LlamaConfig)
        head_mask, intermediate_mask, hidden_mask = self.head_mask, self.intermediate_mask, self.hidden_mask
        if detect_no_zero(head_mask) and detect_no_zero(intermediate_mask) and detect_no_zero(hidden_mask):
            print("No pruning is performed. Skipping pruning.")
            return
        pruned_history = {}
        
        head_mask, intermediate_mask = self.split_mask_or_score(head_mask, intermediate_mask)
        pruned_history['head_mask'] = head_mask
        pruned_history['intermediate_mask'] = intermediate_mask
        pruned_history['hidden_mask'] = self.hidden_mask

        if head_mask is not None:
            decoder_pruned_heads = {}
            pruned_heads = self.config.pruned_heads if continual_pruning else {}
            for layer in range(head_mask.shape[0] if isinstance(head_mask, torch.Tensor) else len(head_mask)):
                if head_mask[layer] is None:
                    continue
                head_to_prune = head_mask[layer] == 0
                if head_to_prune.all():
                    self.head_layer_z[layer] = 0
                now_pruning_heads = torch.where(head_to_prune)[0].tolist()
                if layer in pruned_heads and pruned_heads[layer]:
                    retained_head_indices = [x for x in range(self.config.num_attention_heads) if x not in pruned_heads[layer]]
                    decoder_pruned_heads[layer] = [retained_head_indices[x] for x in now_pruning_heads]
                else:
                    decoder_pruned_heads[layer] = now_pruning_heads
            self.prune_heads(decoder_pruned_heads)
            
        hidden_size = self.lm_head.weight.shape[1]
        if isinstance(hidden_mask, torch.Tensor) and hidden_mask.numel() != hidden_size:
            print("hidden mask's length is not equal to hidden_size, hidden mask's length is", hidden_mask.numel(), "hidden_size is", hidden_size)
            print("Skipping hidden dimension pruning")
        elif hidden_mask is not None and (hidden_mask == 0).any():
            index = torch.LongTensor(hidden_mask.squeeze().nonzero().squeeze().tolist())
            index = index.to(self.device)

            self.model.embed_tokens.weight = torch.nn.parameter.Parameter(
                self.model.embed_tokens.weight.index_select(1, index).detach().clone())
            self.model.embed_tokens.embedding_dim = index.shape[0]
            self.model.norm.weight = torch.nn.parameter.Parameter(
                self.model.norm.weight.index_select(0, index).detach().clone()
            )

            for layer in range(0, self.config.num_hidden_layers):
                print("Pruning layer:", layer)
                if self.model.layers[layer].self_attn.q_proj is not None:
                    self.model.layers[layer].self_attn.q_proj = \
                        prune_layer(self.model.layers[layer].self_attn.q_proj , index, dim=1)
                    self.model.layers[layer].self_attn.k_proj = \
                        prune_layer(self.model.layers[layer].self_attn.k_proj , index, dim=1)
                if self.model.layers[layer].self_attn.v_proj is not None:
                    self.model.layers[layer].self_attn.v_proj = \
                        prune_layer(self.model.layers[layer].self_attn.v_proj , index, dim=1)
                    self.model.layers[layer].self_attn.o_proj = \
                        prune_layer( self.model.layers[layer].self_attn.o_proj , index, dim=0)
                self.model.layers[layer].input_layernorm.weight = nn.Parameter(self.model.layers[layer].input_layernorm.weight.index_select(0, index).detach().clone())
                self.model.layers[layer].post_attention_layernorm.weight = nn.Parameter(self.model.layers[layer].post_attention_layernorm.weight.index_select(0, index).detach().clone())
                    
                if self.model.layers[layer].mlp.up_proj is not None:
                    self.model.layers[layer].mlp.up_proj = \
                        prune_layer( self.model.layers[layer].mlp.up_proj, index, dim=1)
                    self.model.layers[layer].mlp.gate_proj = \
                        prune_layer( self.model.layers[layer].mlp.gate_proj, index, dim=1)
                    self.model.layers[layer].mlp.down_proj = \
                        prune_layer( self.model.layers[layer].mlp.down_proj, index, dim=0)

            self.lm_head = prune_layer(self.lm_head, index, dim=1)

            if getattr(self, "layer_transformation", None) is not None:
                self.layer_transformation = prune_layer(self.layer_transformation, index, dim=1)
                print("layer transformation", self.layer_transformation.weight.shape)
            if getattr(self, "mha_layer_transformation", None) is not None:
                self.mha_layer_transformation = prune_layer(self.mha_layer_transformation, index, dim=1)
                print("layer mha_layer_transformation", self.mha_layer_transformation.weight.shape)
            
        encoder_kept_intermediate_dims = {}
        if intermediate_mask is not None:
            for layer in range(intermediate_mask.shape[0] if isinstance(intermediate_mask, torch.Tensor) else len(intermediate_mask)):
                if intermediate_mask[layer] is None:
                    continue
                intermediate_to_retain = intermediate_mask[layer] != 0
                if not intermediate_to_retain.any():
                    self.mlp_z[layer] = 0
                encoder_kept_intermediate_dims[layer] = torch.where(intermediate_to_retain)[0].tolist()
            self.resize_intermediate(encoder_kept_intermediate_dims)
        self.print_model_shape()
        self.head_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in head_mask])
        self.intermediate_mask = torch.cat([v[v.nonzero().squeeze()].detach().contiguous().flatten() for v in intermediate_mask])
        self.hidden_mask = hidden_mask[hidden_mask.nonzero().squeeze()].detach().contiguous().flatten() if hidden_mask is not None else None
        self.pruned_history = pruned_history
        
    def virtual_prune(self):
        if self.virtual_pruned:
            print("Model is already virtual pruned. Skipping virtual pruning.", flush=True)
            return
        print("Virtual pruning model.", flush=True)
        head_mask, intermediate_mask = self.split_mask_or_score()
        hidden_mask = self.hidden_mask
        hidden_retained_indices = hidden_mask.nonzero().squeeze()
        assert isinstance(self.config, LlamaConfig)
        num_dim_per_head = self.config.hidden_size // self.config.num_attention_heads
        self.retained_indices = hidden_retained_indices
        self.model.retained_indices = hidden_retained_indices
        self.model.norm.retained_indices = hidden_retained_indices
        for layer in range(self.config.num_hidden_layers):
            layer_head_mask, layer_intermediate_mask = head_mask[layer], intermediate_mask[layer]
            
            # Identify retained heads for continual pruning
            decoder_self_pruned_heads = self.config.pruned_heads.get(layer, {})
            retained_head_indices = [x for x in range(self.config.num_attention_heads) if x not in decoder_self_pruned_heads]
            
            # decoder self-mha
            num_retained_heads = layer_head_mask.sum().int().item()
            pruned_heads = (layer_head_mask == 0).nonzero().squeeze()
            pruned_heads = pruned_heads.tolist() if pruned_heads.ndim > 0 else [pruned_heads.item()]
            pruned_heads = [retained_head_indices[v] for v in pruned_heads]
            enc_mha_retained_indices = torch.repeat_interleave(layer_head_mask, num_dim_per_head).nonzero().squeeze()
            
            decoder_layer: ElasticLlamaDecoderLayer = self.model.layers[layer]
            attn_layer: ElasticLlamaAttention = decoder_layer.self_attn
            attn_layer.num_teacher_heads = attn_layer.num_heads
            attn_layer.teacher_pruned_heads = attn_layer.pruned_heads
            attn_layer.num_teacher_key_value_heads = attn_layer.num_key_value_heads
            attn_layer.teacher_hidden_size = attn_layer.hidden_size
            
            attn_layer.pruned_heads = attn_layer.pruned_heads.union(set(pruned_heads))
            attn_layer.num_heads = num_retained_heads
            attn_layer.num_key_value_heads = num_retained_heads
            attn_layer.hidden_size = attn_layer.num_heads * num_dim_per_head
            
            attn_layer.block_retained_indices = enc_mha_retained_indices
            attn_layer.hidden_retained_indices = hidden_retained_indices
            decoder_layer.input_layernorm.retained_indices = hidden_retained_indices
            decoder_layer.post_attention_layernorm.retained_indices = hidden_retained_indices
            
            # decoder ffn
            ffn_retained_indices = layer_intermediate_mask.nonzero().squeeze()
            ffn_layer: ElasticLlamaMLP = decoder_layer.mlp
            ffn_layer.hidden_retained_indices = hidden_retained_indices
            ffn_layer.block_retained_indices = ffn_retained_indices

        self.backup_head_mask = self.head_mask
        self.backup_intermediate_mask = self.intermediate_mask
        self.backup_hidden_mask = self.hidden_mask
        self.head_mask = None
        self.intermediate_mask = None
        self.hidden_mask = None
        self.virtual_pruned = True
            
    def virtual_prune_restore(self):
        if not self.virtual_pruned:
            print("Model is not virtual pruned. Skipping virtual pruning restoration.", flush=True)
            return
        print("Restoring model from virtual pruning", flush=True)
        self.head_mask, self.intermediate_mask, self.hidden_mask = self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask
        self.backup_head_mask, self.backup_intermediate_mask, self.backup_hidden_mask = None, None, None
        self.retained_indices = None
        self.model.retained_indices = None
        self.model.norm.retained_indices = None

        for layer in range(self.config.num_hidden_layers):
            decoder_layer: ElasticLlamaDecoderLayer = self.model.layers[layer]
            attn_layer: ElasticLlamaAttention = decoder_layer.self_attn
            # decoder self-mha
            attn_layer.num_heads = attn_layer.num_teacher_heads
            attn_layer.num_key_value_heads = attn_layer.num_teacher_key_value_heads
            attn_layer.hidden_size = attn_layer.teacher_hidden_size
            attn_layer.pruned_heads = attn_layer.teacher_pruned_heads
            attn_layer.block_retained_indices = None
            attn_layer.hidden_retained_indices = None
            decoder_layer.input_layernorm.retained_indices = None
            decoder_layer.post_attention_layernorm.retained_indices = None
            
            # decoder ffn
            ffn_layer: ElasticLlamaMLP = decoder_layer.mlp
            ffn_layer.hidden_retained_indices = None
            ffn_layer.block_retained_indices = None

        self.virtual_pruned = False
        
    def resize_intermediate(self, kept_intermediate_dims: Dict[int, List[int]]):
        model = self.model
        device = self.device
        for layer in kept_intermediate_dims:
            if len(kept_intermediate_dims[layer]) == 0:
                model.layers[layer].mlp.up_proj = None
                model.layers[layer].mlp.gate_proj = None
                model.layers[layer].mlp.down_proj = None
            else:
                model.layers[layer].mlp.gate_proj = prune_layer(model.layers[layer].mlp.gate_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
                model.layers[layer].mlp.up_proj = prune_layer(model.layers[layer].mlp.up_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=0)
                model.layers[layer].mlp.down_proj = prune_layer(model.layers[layer].mlp.down_proj, index=torch.LongTensor(kept_intermediate_dims[layer]).to(device), dim=1)
            
    def print_model_shape(self):
        for layer in range(self.config.num_hidden_layers):
            print("Layer:", layer)
            if self.model.layers[layer].self_attn.q_proj is not None:
                print("self-attention query:", self.model.layers[layer].self_attn.q_proj.weight.shape)
                print("self-attention key:", self.model.layers[layer].self_attn.k_proj.weight.shape)
            else:
                print("self-attention query:", None)
                print("self-attention key:", None)
            if self.model.layers[layer].self_attn.v_proj is not None:
                print("self-attention value:", self.model.layers[layer].self_attn.v_proj.weight.shape)
                print("self-attention output:", self.model.layers[layer].self_attn.o_proj.weight.shape)
            else:
                print("self-attention value:", None)
                print("self-attention output:", None)
            wi = self.model.layers[layer].mlp.gate_proj
            if wi is not None:
                print("up & gated:", wi.weight.shape)
                print("down:", self.model.layers[layer].mlp.down_proj.weight.shape)
            else:
                print("up & gated", None)
                print("down", None)
    
    def print_lora_info_by_layer(self):
        def print_lora_info(l, layername):
            if isinstance(l, lora.Linear) and hasattr(l, 'lora_A') and hasattr(l, 'lora_B') and l.lora_A is not None and l.lora_B is not None:
                print("%s: r: " % layername, l.r if hasattr(l, 'r') else 0, ', input dim: ', l.lora_A.shape[1] if hasattr(l, 'lora_A') and l.lora_A is not None else 0, ', output dim: ', l.lora_B.shape[0] if hasattr(l, 'lora_B') and l.lora_B is not None else 0)
            elif isinstance(l, lora.Linear):
                print("%s: frozen LoRA layer" % layername)
            else:
                print("%s: frozen Linear layer" % layername)
        for i in range(self.config.num_hidden_layers):
            print("Layer:", i)
            layer: ElasticLlamaDecoderLayer = self.model.layers[i]
            query, key, value, output = layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj
            up, gate, down = layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj
            print_lora_info(query, "query")
            print_lora_info(key, "key")
            print_lora_info(value, "value")
            print_lora_info(output, "output")
            print_lora_info(up, "up")
            print_lora_info(gate, "gate")
            print_lora_info(down, "down")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_teacher: bool =False,
        head_z: Optional[torch.Tensor]=None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        pass_mask: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        head_z = head_z if head_z is not None else self.head_mask if pass_mask else None
        intermediate_z = intermediate_z if intermediate_z is not None else self.intermediate_mask if pass_mask else None
        hidden_z = hidden_z if hidden_z is not None else self.hidden_mask if pass_mask else None
        # Using bottom-up pruning, disable layer-level zs
        head_layer_z = None
        mlp_z = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        head_z, intermediate_z = split_mask_or_score(self, head_z, intermediate_z)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_teacher=use_teacher,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if isinstance(self.lm_head, lora.PruningLinear):
                logits = self.lm_head(hidden_states, use_teacher=use_teacher, in_retained_indices=self.retained_indices)
            elif use_teacher:
                logits = self.lm_head(hidden_states)
            else:
                selected_weight, selected_bias = select_wandb(self.lm_head.weight, self.lm_head.bias, in_retained_indices=self.retained_indices)
                logits = F.linear(hidden_states, selected_weight, selected_bias)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Helper functions (imported from other files)
def detect_no_zero(mask):
    if isinstance(mask, torch.Tensor):
        return (mask == 1).all().item()
    elif isinstance(mask, list):
        return all((m == 1).all().item() if isinstance(m, torch.Tensor) else True for m in mask)
    return True

def split_mask_or_score(model, head_mask, intermediate_mask):
    # Implementation of split_mask_or_score function
    # This is a simplified version - actual implementation would be more complex
    if isinstance(head_mask, torch.Tensor):
        head_mask = head_mask.view(model.config.num_hidden_layers, model.config.num_attention_heads)
    if isinstance(intermediate_mask, torch.Tensor):
        intermediate_mask = intermediate_mask.view(model.config.num_hidden_layers, -1)
    
    head_mask_list = []
    intermediate_mask_list = []
    
    if isinstance(head_mask, torch.Tensor):
        for i in range(model.config.num_hidden_layers):
            head_mask_list.append(head_mask[i])
    else:
        head_mask_list = head_mask
        
    if isinstance(intermediate_mask, torch.Tensor):
        for i in range(model.config.num_hidden_layers):
            intermediate_mask_list.append(intermediate_mask[i])
    else:
        intermediate_mask_list = intermediate_mask
        
    return head_mask_list, intermediate_mask_list

def prune_layer(layer, index, dim):
    # Implementation of prune_layer function
    if dim == 0:
        new_weight = layer.weight.index_select(0, index)
        if layer.bias is not None:
            new_bias = layer.bias.index_select(0, index)
        else:
            new_bias = None
    elif dim == 1:
        new_weight = layer.weight.index_select(1, index)
        new_bias = layer.bias
    else:
        raise ValueError("dim must be 0 or 1")
    
    new_layer = torch.nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=layer.bias is not None)
    new_layer.weight.data = new_weight
    if new_bias is not None:
        new_layer.bias.data = new_bias
    return new_layer

# Import lora module
try:
    import loralib as lora
except ImportError:
    # Create a minimal lora implementation if not available
    class Linear(nn.Module):
        def __init__(self, in_features, out_features, r=8, lora_alpha=16, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.r = r
            self.lora_alpha = lora_alpha
            self.lora_dropout = nn.Dropout(0.05)
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias = None
            self.reset_parameters()
            
        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            
        def forward(self, x, use_teacher=False):
            if use_teacher:
                return F.linear(x, self.weight, self.bias)
            else:
                result = F.linear(x, self.weight, self.bias)
                result += (self.lora_B(self.lora_A(self.lora_dropout(x)))) * (self.lora_alpha / self.r)
                return result
    lora = type('lora', (), {'Linear': Linear})()