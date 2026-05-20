#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluation script for Llama models on MMLU dataset.
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

def load_mmlu_data(data_dir, ntrain=5):
    """Load MMLU dataset"""
    subjects = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]
    
    data = {}
    for subject in subjects:
        data[subject] = load_dataset("lukaemon/mmlu", subject, split="test")
    
    return data, subjects

def format_example(question, options, ntrain=5):
    """Format example for prompting"""
    prompt = "The following are multiple choice questions (with answers).\n\n"
    prompt += question + "\n"
    for i, option in enumerate(options):
        prompt += f"{chr(65+i)}. {option}\n"
    prompt += "Answer:"
    return prompt

def evaluate_model(model, tokenizer, data, subjects, ntrain=5, batch_size=2):
    """Evaluate model on MMLU dataset"""
    results = {}
    total_correct = 0
    total_count = 0
    
    for subject in subjects:
        print(f"Evaluating {subject}...")
        subject_data = data[subject]
        correct = 0
        count = 0
        
        # Process in batches
        for i in range(0, len(subject_data), batch_size):
            batch = subject_data[i:i+batch_size]
            prompts = []
            answers = []
            
            for example in batch:
                question = example['question']
                options = [example['A'], example['B'], example['C'], example['D']]
                prompt = format_example(question, options, ntrain)
                prompts.append(prompt)
                answers.append(example['answer'])
            
            # Tokenize prompts
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate predictions
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract predictions
            predictions = []
            for i, output in enumerate(outputs):
                # Get the last token
                last_token = output[len(inputs['input_ids'][i]):][0]
                pred = tokenizer.decode(last_token, skip_special_tokens=True)
                predictions.append(pred)
            
            # Calculate accuracy
            for pred, answer in zip(predictions, answers):
                if pred.strip().upper() == answer.strip().upper():
                    correct += 1
                count += 1
            
            total_correct += correct
            total_count += count
        
        results[subject] = correct / count if count > 0 else 0
        print(f"  Accuracy: {results[subject]:.4f}")
    
    results['average'] = total_correct / total_count if total_count > 0 else 0
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    args = parser.parse_args()
    
    # Load model and tokenizer
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load data
    data, subjects = load_mmlu_data(args.data_dir, args.ntrain)
    
    # Evaluate
    results = evaluate_model(model, tokenizer, data, subjects, args.ntrain, args.eval_batch_size)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "mmlu_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("MMLU RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name_or_path}")
    print(f"Average Accuracy: {results['average']:.4f}")
    print("-"*50)
    
    for subject in sorted(subjects):
        print(f"{subject}: {results[subject]:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()