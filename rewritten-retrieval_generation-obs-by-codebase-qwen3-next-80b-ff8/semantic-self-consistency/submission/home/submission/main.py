import os
import json
import random
import argparse
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from collections import Counter
import csv

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Cosine similarity implementation as defined in paper
def cosine_similarity_vectors(a, b):
    """
    Implements cosine similarity: (a·b) / (||a||_2 · ||b||_2)
    """
    # Convert to torch tensors
    a_tensor = torch.tensor(a, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    
    # Compute cosine similarity
    sim = cosine_similarity(a_tensor.unsqueeze(0), b_tensor.unsqueeze(0), dim=1)
    return sim.item()

# Semantic weighting function as described in paper cards
def compute_semantic_weights(rationale_embeddings, threshold=0.3):
    """
    Compute semantic weights for rationales using cosine similarity.
    - Uses cosine similarity between all pairs of embeddings
    - Applies threshold of 0.3 (minimum similarity)
    - Normalizes weights by L2 norm
    """
    n = len(rationale_embeddings)
    if n == 0:
        return []
    
    # Compute pairwise cosine similarities
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarities[i][j] = 1.0
            else:
                sim = cosine_similarity_vectors(rationale_embeddings[i], rationale_embeddings[j])
                similarities[i][j] = max(sim, threshold)  # Apply minimum threshold
    
    # Compute weights for each rationale (sum of similarities to all others)
    weights = np.sum(similarities, axis=1)
    
    # Normalize by L2 norm (as specified in paper_card_0003)
    l2_norm = np.linalg.norm(weights)
    if l2_norm > 0:
        weights = weights / l2_norm
    
    return weights.tolist()

# Majority vote with semantic weights
def weighted_majority_vote(answers, weights, tie_breaker='random'):
    """
    Perform majority vote on final answers with semantic weights.
    - Input: list of answers and corresponding semantic weights
    - Selects most frequent answer weighted by semantic weights
    - Ties resolved by uniform random selection (paper_card_0014)
    """
    if len(answers) == 0:
        return None
    
    # Create weighted counter
    weighted_counts = {}
    for answer, weight in zip(answers, weights):
        if answer not in weighted_counts:
            weighted_counts[answer] = 0
        weighted_counts[answer] += weight
    
    # Find the answer with maximum weight
    max_weight = max(weighted_counts.values())
    best_answers = [answer for answer, weight in weighted_counts.items() if weight == max_weight]
    
    # Resolve ties with uniform random selection
    if len(best_answers) > 1 and tie_breaker == 'random':
        return random.choice(best_answers)
    else:
        return best_answers[0]

# Load and preprocess datasets
def load_dataset_split(dataset_name, split="test", num_samples=None):
    """
    Load dataset splits with appropriate preprocessing
    """
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        if split not in dataset:
            raise ValueError(f"Split {split} not available in GSM8K")
        examples = dataset[split]
        # Format: {"question": "...", "answer": "#### 123"}
        processed = []
        for ex in examples:
            question = ex["question"]
            # Extract answer (everything after ####)
            answer_text = ex["answer"].split("####")[-1].strip()
            processed.append({
                "question": question,
                "answer": answer_text
            })
        return processed[:num_samples] if num_samples else processed
    
    elif dataset_name == "math":
        dataset = load_dataset("math_dataset", "mathqa")
        if split not in dataset:
            raise ValueError(f"Split {split} not available in MATH")
        examples = dataset[split]
        processed = []
        for ex in examples:
            question = ex["question"]
            # Extract answer (usually in format: \boxed{answer})
            answer_text = ex["answer"].replace("\\boxed{", "").replace("}", "").strip()
            processed.append({
                "question": question,
                "answer": answer_text
            })
        return processed[:num_samples] if num_samples else processed
    
    elif dataset_name == "svamp":
        # Load SVAMP from local file
        with open("/tmp/datasets/SVAMP.json", "r") as f:
            data = json.load(f)
        processed = []
        for ex in data:
            question = ex["Question"]
            answer = str(ex["Answer"])
            processed.append({
                "question": question,
                "answer": answer
            })
        return processed[:num_samples] if num_samples else processed
    
    elif dataset_name == "asdiv":
        # ASDIV dataset from Hugging Face
        dataset = load_dataset("ai2_arc", "ASDiv")
        if split not in dataset:
            raise ValueError(f"Split {split} not available in ASDIV")
        examples = dataset[split]
        processed = []
        for ex in examples:
            question = ex["question"]
            answer = ex["answerKey"]
            processed.append({
                "question": question,
                "answer": answer
            })
        return processed[:num_samples] if num_samples else processed
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Generate rationale using chain-of-thought prompting
def generate_rationale(model, tokenizer, question, max_new_tokens=250, temperature=0.7, top_p=0.9):
    """
    Generate a reasoning path using chain-of-thought prompting
    """
    prompt = f"""Question: {question}
Let's think step by step:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output and extract rationale
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    rationale = output_text[len(prompt):].strip()
    return rationale

# Extract final answer from rationale
def extract_final_answer(rationale):
    """
    Extract final answer from rationale using common patterns
    """
    # Common patterns for final answers
    patterns = [
        r"####\s*([0-9.-]+)",
        r"\boxed{([0-9.-]+)}",
        r"answer is\s*([0-9.-]+)",
        r"so the answer is\s*([0-9.-]+)",
        r"therefore\s*([0-9.-]+)",
        r"final answer:\s*([0-9.-]+)"
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, rationale, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: look for any number at the end
    numbers = re.findall(r"[-+]?\d*\.?\d+", rationale)
    if numbers:
        return numbers[-1]
    
    # If no number found, return empty string
    return ""

# Generate semantic embeddings for rationales
def generate_embeddings(rationales, embedding_model, tokenizer):
    """
    Generate semantic embeddings for rationales using sentence transformer
    """
    # Use a lightweight embedding model for efficiency
    inputs = tokenizer(rationales, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

# Main evaluation function
def evaluate_semantic_self_consistency(model, tokenizer, embedding_model, embedding_tokenizer, 
                                     dataset_name, examples, num_rationales=5, output_dir="results"):
    """
    Evaluate semantic self-consistency on a dataset
    """
    print(f"Evaluating {dataset_name} with {num_rationales} rationales per question...")
    
    correct = 0
    total = 0
    results = []
    
    start_time = time.time()
    
    for i, example in enumerate(examples):
        question = example["question"]
        true_answer = example["answer"]
        
        # Generate multiple rationales
        rationales = []
        final_answers = []
        
        for _ in range(num_rationales):
            rationale = generate_rationale(model, tokenizer, question)
            rationales.append(rationale)
            final_answer = extract_final_answer(rationale)
            final_answers.append(final_answer)
        
        # Generate embeddings for rationales
        if len(rationales) > 0:
            embeddings = generate_embeddings(rationales, embedding_model, embedding_tokenizer)
            
            # Compute semantic weights
            weights = compute_semantic_weights(embeddings, threshold=0.3)
            
            # Perform weighted majority vote
            predicted_answer = weighted_majority_vote(final_answers, weights, tie_breaker='random')
        else:
            predicted_answer = ""
        
        # Check if correct
        is_correct = (predicted_answer == true_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # Store results
        results.append({
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "rationales": rationales,
            "final_answers": final_answers,
            "weights": weights
        })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(examples)} examples, accuracy: {correct/total:.3f}")
    
    accuracy = correct / total if total > 0 else 0
    total_time = time.time() - start_time
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{dataset_name}_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "accuracy", "total_examples", "num_rationales", "total_time_seconds"])
        writer.writerow([dataset_name, accuracy, total, num_rationales, total_time])
    
    print(f"{dataset_name} Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Total time: {total_time:.2f} seconds")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math", "svamp", "asdiv"],
                       help="Datasets to evaluate on")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples to evaluate (paper uses 1000)")
    parser.add_argument("--num_rationales", type=int, default=5,
                       help="Number of rationales to generate per question")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="Model to use for reasoning")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Model to use for semantic embeddings")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load embedding model
    print(f"Loading embedding model: {args.embedding_model_name}")
    embedding_model = AutoModelForCausalLM.from_pretrained(
        args.embedding_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    
    # Ensure model is in eval mode
    model.eval()
    embedding_model.eval()
    
    # Load datasets
    all_accuracies = {}
    
    for dataset_name in args.datasets:
        print(f"\nLoading {dataset_name} dataset...")
        examples = load_dataset_split(dataset_name, "test", args.num_samples)
        print(f"Loaded {len(examples)} examples from {dataset_name}")
        
        # Evaluate semantic self-consistency
        accuracy = evaluate_semantic_self_consistency(
            model, tokenizer, embedding_model, embedding_tokenizer,
            dataset_name, examples, args.num_rationales, args.output_dir
        )
        all_accuracies[dataset_name] = accuracy
    
    # Save overall summary
    summary_file = os.path.join(args.output_dir, "overall_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "accuracy"])
        for dataset, acc in all_accuracies.items():
            writer.writerow([dataset, acc])
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for dataset, acc in all_accuracies.items():
        print(f"{dataset}: {acc:.3f}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()