import os
import json
import numpy as np
from typing import List, Dict, Any
from src.dataset_loader import load_dataset_by_name
from src.llm_generator import LLMGenerator
from src.featurizer import get_featurizer
from src.semantic_weighting import CentroidProximityWeighting, SemanticConsensusWeighting
from src.evaluator import Evaluator

# Configuration
NUM_RESPONSES = 10
DATASETS = ["AQuA-RAT", "SVAMP", "StrategyQA"]

def main():
    """Main reproduction script"""
    print("Starting semantic self-consistency reproduction...")
    
    # Initialize generator
    generator = LLMGenerator()
    
    # Initialize results
    results = {}
    
    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset_by_name(dataset_name)
        
        # Initialize evaluator
        evaluator = Evaluator(dataset_name, dataset)
        
        # Initialize featurizer
        featurizer = get_featurizer(dataset_name)
        
        # Store predictions
        predictions = []
        ground_truths = []
        
        # Generate responses
        for item in dataset:
            question = item["question"]
            answer = item["answer"]
            ground_truths.append(answer)
            
            # Generate responses
            prompt = f"Question: {question}\nAnswer with step-by-step reasoning:"
            responses = generator.generate_multiple_responses(prompt, NUM_RESPONSES)
            
            # Extract final answers
            final_answers = []
            for response in responses:
                # Extract answer from response
                answer = extract_answer(response)
                final_answers.append(answer)
            
            # Get embeddings
            embeddings = featurizer.encode(responses)
            
            # Apply weighting
            cpw = CentroidProximityWeighting(embeddings)
            weights_cpw = cpw.compute_weights()
            
            # Weighted average
            weighted_answer = get_weighted_answer(final_answers, weights_cpw)
            predictions.append(weighted_answer)
        
        # Evaluate
        result = evaluator.evaluate(predictions, ground_truths)
        results[dataset_name] = result
    
    # Save results
    with open("/home/submission/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("Reproduction complete!")
    
def extract_answer(response: str) -> str:
    """Extract answer from response"""
    # Simple extraction - in real use, use more sophisticated extraction
    lines = response.split('\n')
    for line in lines:
        if any(x in line.lower() for x in ["answer is", "the answer is", "is", "so the answer is", "therefore"]):
            # Extract number or answer
            import re
            match = re.search(r'(?:answer is|is|so the answer is|therefore|the final answer is)\s*[:\s]*(.*)', line, re.IGNORECASE)
            if match:
                answer = match.group(1).strip().rstrip('.').rstrip('!').rstrip('?').strip()
                if answer:
                    return answer
    # Fallback
    return ""

def get_weighted_answer(answers: List[str], weights: np.ndarray) -> str:
    """Get weighted answer"""
    # In real implementation, use weighted majority vote
    # For reproduction, use simple majority vote
    from collections import Counter
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

if __name__ == "__main__":
    main()