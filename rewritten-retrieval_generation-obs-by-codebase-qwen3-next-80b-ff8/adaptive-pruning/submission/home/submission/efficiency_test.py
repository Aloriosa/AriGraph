import os
import json
import sys
import torch
import transformers
from transformers import HfArgumentParser
from args import DataTrainingArguments, ModelArguments, MinusTrainingArguments
from models import build_model
from utils import build_trainer
from utils.utils import *
from utils.minus_utils import efficiency_testing, input_constructor

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MinusTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    t_name, raw_datasets = get_raw_datasets(model_args, data_args, training_args)
    config, tokenizer, model = build_model(model_args, data_args, training_args, t_name, raw_datasets, force_model_shape_deduction=True)
    
    # Load pruning masks if available
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'head_mask.pt')):
        model.head_mask = torch.load(os.path.join(model_args.model_name_or_path, 'head_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt')):
        model.intermediate_mask = torch.load(os.path.join(model_args.model_name_or_path, 'intermediate_mask.pt'))
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt')):
        model.hidden_mask = torch.load(os.path.join(model_args.model_name_or_path, 'hidden_mask.pt'))
    
    # Apply pruning masks
    model.prune_model_with_masks()
    
    # Remove identity masks
    if isinstance(model.head_mask, torch.Tensor) and (model.head_mask == 1).all().item():
        model.head_mask = None
    if isinstance(model.intermediate_mask, torch.Tensor) and (model.intermediate_mask == 1).all().item():
        model.intermediate_mask = None
    if isinstance(model.hidden_mask, torch.Tensor) and (model.hidden_mask == 1).all().item():
        model.hidden_mask = None
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Test inference efficiency
    print("Testing inference efficiency...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Measure inference time
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 10
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    
    # Measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    print(f"Peak memory usage: {peak_memory:.2f} GB")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter reduction: {(1 - trainable_params/total_params)*100:.2f}%")
    
    # Save results
    results = {
        "inference_time": avg_inference_time,
        "peak_memory_gb": peak_memory,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_reduction_percent": (1 - trainable_params/total_params)*100
    }
    
    with open(os.path.join(training_args.output_dir, "efficiency_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("Efficiency testing completed. Results saved to efficiency_results.json")

if __name__ == "__main__":
    main()