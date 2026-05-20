#!/usr/bin/env python3
"""
Semantic Self-Consistency Implementation
Implements the method described in the paper: 
- Samples multiple rationales
- Computes semantic embeddings
- Applies semantic weighting
- Uses majority vote with weights
"""

import os
import json
import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import argparse
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSelfConsistency:
    def __init__(self, 
                 model_name: str = "gpt2",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 num_samples: int = 5,
                 min_similarity: float = 0.3,
                 max_new_tokens: int = 250,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Semantic Self-Consistency system
        
        Args:
            model_name: Name of the LLM to use
            embedding_model_name: Name of the embedding model
            num_samples: Number of rationale samples to generate
            min_similarity: Minimum similarity threshold for weighting
            max_new_tokens: Maximum tokens to generate for responses
            device: Device to run models on
        """
        self.num_samples = num_samples
        self.min_similarity = min_similarity
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        # Load LLM
        logger.info(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm.to(device)
        self.llm.eval()
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(device)
        
        # Define prompt templates for different datasets
        self.prompt_templates = {
            "gsm8k": "Question: {question}\nLet's think step by step:",
            "math": "Question: {question}\nLet's think step by step:",
            "svamp": "Question: {question}\nLet's think step by step:",
            "asdiv": "Question: {question}\nLet's think step by step:"
        }
        
        # Extract final answer from response
        self.answer_patterns = [
            "the answer is",
            "answer is",
            "final answer is",
            "so the answer is",
            "therefore the answer is"
        ]
    
    def extract_final_answer(self, response: str) -> str:
        """
        Extract final numerical answer from LLM response
        """
        response_lower = response.lower()
        
        # Look for answer patterns
        for pattern in self.answer_patterns:
            if pattern in response_lower:
                # Extract the part after the pattern
                start_idx = response_lower.find(pattern) + len(pattern)
                answer_part = response[start_idx:].strip()
                
                # Extract the first number-like string
                import re
                # Look for numbers (integers, decimals, fractions)
                numbers = re.findall(r'[-+]?\d*\.?\d+|[-+]?\d+/\d+', answer_part)
                if numbers:
                    return numbers[0]
        
        # If no pattern found, try to find any number in the response
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+|[-+]?\d+/\d+', response)
        if numbers:
            return numbers[-1]  # Take the last number as the final answer
        
        # If no number found, return empty string
        return ""
    
    def generate_rationale(self, question: str, dataset: str) -> str:
        """
        Generate a single rationale for the question using the LLM
        """
        prompt_template = self.prompt_templates.get(dataset, "{question}")
        prompt = prompt_template.format(question=question)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):]
        
        return response.strip()
    
    def get_semantic_embeddings(self, rationales: List[str]) -> np.ndarray:
        """
        Get semantic embeddings for a list of rationales
        """
        embeddings = self.embedding_model.encode(rationales, convert_to_numpy=True)
        return embeddings
    
    def compute_cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        # Normalize embeddings
        embedding_a_norm = embedding_a / np.linalg.norm(embedding_a)
        embedding_b_norm = embedding_b / np.linalg.norm(embedding_b)
        
        # Compute cosine similarity
        similarity = np.dot(embedding_a_norm, embedding_b_norm)
        return float(similarity)
    
    def compute_semantic_weights(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute semantic weights for each rationale based on cosine similarity
        Implements semantic weighting from paper:
        - Compute pairwise cosine similarities
        - Weight = sum of similarities above threshold
        - Normalize by L2 norm
        """
        n = len(embeddings)
        weights = np.zeros(n)
        
        for i in range(n):
            total_similarity = 0.0
            for j in range(n):
                if i != j:
                    similarity = self.compute_cosine_similarity(embeddings[i], embeddings[j])
                    if similarity >= self.min_similarity:
                        total_similarity += similarity
            
            weights[i] = total_similarity
        
        # Apply L2 normalization
        if np.linalg.norm(weights) > 0:
            weights = weights / np.linalg.norm(weights)
        
        # Ensure non-negative weights
        weights = np.maximum(weights, 0)
        
        return weights
    
    def majority_vote_with_weights(self, answers: List[str], weights: np.ndarray) -> str:
        """
        Perform majority vote with semantic weights
        - Select most frequent answer
        - In case of tie, use weighted majority
        - If still tied, random selection
        """
        if len(answers) == 0:
            return ""
        
        # Count frequency of each answer
        answer_counts = {}
        answer_weights = {}
        
        for i, answer in enumerate(answers):
            if answer not in answer_counts:
                answer_counts[answer] = 0
                answer_weights[answer] = 0.0
            answer_counts[answer] += 1
            answer_weights[answer] += weights[i]
        
        # Find answers with maximum count
        max_count = max(answer_counts.values())
        best_answers = [answer for answer, count in answer_counts.items() if count == max_count]
        
        if len(best_answers) == 1:
            return best_answers[0]
        
        # Tie-break by weight
        best_answer_by_weight = max(best_answers, key=lambda x: answer_weights[x])
        
        # If still tied, random selection
        best_weights = [answer_weights[answer] for answer in best_answers]
        max_weight = max(best_weights)
        final_best_answers = [answer for answer in best_answers if answer_weights[answer] == max_weight]
        
        return random.choice(final_best_answers)
    
    def evaluate_dataset(self, dataset_path: str, dataset_name: str) -> Dict:
        """
        Evaluate on a single dataset
        """
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        correct = 0
        total = 0
        results = []
        
        for item in tqdm(data, desc=f"Evaluating {dataset_name}"):
            question = item['question']
            true_answer = item['answer']
            
            # Generate multiple rationales
            rationales = []
            final_answers = []
            
            for _ in range(self.num_samples):
                rationale = self.generate_rationale(question, dataset_name)
                rationales.append(rationale)
                
                # Extract final answer from rationale
                answer = self.extract_final_answer(rationale)
                final_answers.append(answer)
            
            # Compute semantic embeddings
            embeddings = self.get_semantic_embeddings(rationales)
            
            # Compute semantic weights
            weights = self.compute_semantic_weights(embeddings)
            
            # Perform majority vote with weights
            predicted_answer = self.majority_vote_with_weights(final_answers, weights)
            
            # Check if correct
            is_correct = (predicted_answer == true_answer)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "rationales": rationales,
                "final_answers": final_answers,
                "weights": weights.tolist()
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def evaluate_all_datasets(self, datasets: Dict[str, str]) -> Dict:
        """
        Evaluate on all datasets
        """
        results = {}
        overall_correct = 0
        overall_total = 0
        
        for dataset_name, dataset_path in datasets.items():
            dataset_results = self.evaluate_dataset(dataset_path, dataset_name)
            results[dataset_name] = dataset_results
            overall_correct += dataset_results["correct"]
            overall_total += dataset_results["total"]
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        
        return {
            "per_dataset": results,
            "overall_accuracy": overall_accuracy,
            "correct": overall_correct,
            "total": overall_total
        }

def main():
    parser = argparse.ArgumentParser(description='Semantic Self-Consistency Evaluation')
    parser.add_argument('--model', type=str, default='gpt2', help='LLM model name')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of rationale samples')
    parser.add_argument('--min-similarity', type=float, default=0.3, help='Minimum similarity threshold')
    parser.add_argument('--max-new-tokens', type=int, default=250, help='Maximum new tokens to generate')
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Define datasets
    datasets = {
        "gsm8k": "/home/submission/data/gsm8k_sample.json",
        "math": "/home/submission/data/math_sample.json",
        "svamp": "/home/submission/data/svamp_sample.json",
        "asdiv": "/home/submission/data/asdiv_sample.json"
    }
    
    # Initialize Semantic Self-Consistency system
    ssc = SemanticSelfConsistency(
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        num_samples=args.num_samples,
        min_similarity=args.min_similarity,
        max_new_tokens=args.max_new_tokens
    )
    
    # Evaluate on all datasets
    results = ssc.evaluate_all_datasets(datasets)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SEMANTIC SELF-CONSISTENCY RESULTS")
    print("="*60)
    
    for dataset_name, result in results["per_dataset"].items():
        print(f"{dataset_name.upper()}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    
    print(f"OVERALL: {results['overall_accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    # Save CSV summary
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Accuracy", "Correct", "Total"])
        for dataset_name, result in results["per_dataset"].items():
            writer.writerow([dataset_name, result['accuracy'], result['correct'], result['total']])
        writer.writerow(["Overall", results['overall_accuracy'], results['correct'], results['total']])
    
    print(f"\nResults saved to {args.output} and results.csv")

if __name__ == "__main__":
    main()