import numpy as np
import pandas as pd
import random
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_aqua_rat_dataset(size=254):
    """
    Generate simulated AQuA-RAT dataset similar to paper
    """
    print("Generating AQuA-RAT dataset...")
    
    questions = []
    answers = []
    rationales = []
    
    # Define patterns for arithmetic problems
    patterns = [
        # Basic addition
        {"pattern": "There are {a} apples and {b} oranges. How many fruits total?",
         "answer": lambda a, b: a + b, "vars": ["a", "b"], "type": "addition"},
        # Subtraction
        {"pattern": "I had {a} cookies. I ate {b}. How many left?",
         "answer": lambda a, b: a - b, "vars": ["a", "b"], "type": "subtraction"},
        # Multiplication
        {"pattern": "Each box has {a} candies. There are {b} boxes. How many total?",
         "answer": lambda a, b: a * b, "vars": ["a", "b"], "type": "multiplication"},
        # Division
        {"pattern": "I have {a} candies to share among {b} friends. How many each?",
         "answer": lambda a, b: a // b, "vars": ["a", "b"], "type": "division"},
        # Mixed operations
        {"pattern": "I have {a} books. I bought {b} more. Then I gave away {c}. How many?",
         "answer": lambda a, b, c: a + b - c, "vars": ["a", "b", "c"], "type": "mixed"},
    ]
    
    for i in range(size):
        # Randomly select pattern
        pattern = random.choice(patterns)
        vars_needed = pattern["vars"]
        
        # Generate random values based on pattern
        if len(vars_needed) == 2:
            a, b = random.randint(1, 10), random.randint(1, 10)
            if pattern["type"] == "division":
                # Ensure divisible
                b = random.randint(1, 10)
                a = b * random.randint(1, 10)
            else:
                a, b = random.randint(1, 10), random.randint(1, 10)
            
            if len(vars_needed) == 2:
                answer = pattern["answer"](a, b)
            else:
                c = random.randint(1, 10)
                answer = pattern["answer"](a, b, c)
        
        # Generate rationale
        if pattern["type"] == "addition":
            rationale = f"Step 1: We start with {a} apples. Step 2: We add {b} oranges. Step 3: Total is {a} + {b} = {answer}"
        elif pattern["type"] == "subtraction":
            rationale = f"Step 1: I had {a} cookies. Step 2: I ate {b}. Step 3: Left: {a} - {b} = {answer}"
        elif pattern["type"] == "multiplication":
            rationale = f"Step 1: Each box has {a} candies. Step 2: {b} boxes. Step 3: Total: {a} * {b} = {answer}"
        elif pattern["type"] == "division":
            rationale = f"Step 1: I have {a} candies. Step 2: Share among {b} friends. Step 3: Each gets: {a} / {b} = {answer}"
        elif pattern["type"] == "mixed":
            c = random.randint(1, 5)
            answer = pattern["answer"](a, b, c)
            rationale = f"Step 1: I had {a} books. Step 2: Bought {b} more. Step 3: Gave away {c}. Step 4: Left: {a} + {b} - {c} = {answer}"
        
        # Add to dataset
        questions.append(pattern["pattern"].format(a=a, b=b))
        answers.append(str(answer))
        rationales.append(rationale)
    
    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "rationale": rationales
    })
    
    df.to_csv("data/aqua_rat.csv", index=False)
    print(f"Generated AQuA-RAT dataset with {len(df)} samples")
    return df

def generate_svamp_dataset(size=1000):
    """
    Generate simulated SVAMP dataset similar to paper
    """
    print("Generating SVAMP dataset...")
    
    questions = []
    answers = []
    rationales = []
    
    # Define patterns for algebraic problems
    patterns = [
        # Linear equations
        {"pattern": "Solve for x: {a}x + {b} = {c}",
         "answer": lambda a, b, c: (c - b) / a, "vars": ["a", "b", "c"], "type": "linear"},
        # Quadratic equations
        {"pattern": "Solve for x: x^2 + {a}x + {b} = 0",
         "answer": lambda a, b: (-a + np.sqrt(a**2 - 4*b))/2, "vars": ["a", "b"], "type": "quadratic"},
        # Systems of equations
        {"pattern": "Solve the system: {a}x + {b}y = {c}, {d}x + {e}y = {f}",
         "answer": lambda a, b, c, d, e, f: (c - b*(f-d*c)/(e-a*f)) / (a - b), "vars": ["a", "b", "c", "d", "e", "f"], "type": "system"},
    ]
    
    for i in range(size):
        # Randomly select pattern
        pattern = random.choice(patterns)
        vars_needed = pattern["vars"]
        
        # Generate random values based on pattern
        if pattern["type"] == "linear":
            a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
            answer = pattern["answer"](a, b, c)
        elif pattern["type"] == "quadratic":
            a, b = random.randint(1, 10), random.randint(1, 10)
            answer = pattern["answer"](a, b)
        elif pattern["type"] == "system":
            a, b, c, d, e, f = [random.randint(1, 10) for _ in range(7)]
            answer = pattern["answer"](a, b, c, d, e, f)
        
        # Generate rationale
        if pattern["type"] == "linear":
            rationale = f"Step 1: We have the equation {a}x + {b} = {c}. Step 2: Subtract {b} from both sides: {a}x = {c - b}. Step 3: Divide by {a}: x = {answer}"
        elif pattern["type"] == "quadratic":
            rationale = f"Step 1: We have the equation x^2 + {a}x + {b} = 0. Step 2: Use quadratic formula: x = (-{a} ± √({a}^2 - 4*{b})) / 2. Step 3: x = {answer}"
        elif pattern["type"] == "system":
            rationale = f"Step 1: We have the system: {a}x + {b}y = {c}, {d}x + {e}y = {f}. Step 2: Multiply first equation by {d}, second by {a}: {a*d}x + {b*d}y = {c*d}, {a*d}x + {a*e}y = {a*f}. Step 3: Subtract: ({a*e} - {a*d})y = {a*f} - {a*c}. Step 4: y = {answer}"
        
        # Add to dataset
        questions.append(pattern["pattern"].format(a=a, b=b, c=c))
        answers.append(str(round(answer, 2)))
        rationales.append(rationale)
    
    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "rationale": rationales
    })
    
    df.to_csv("data/svamp.csv", index=False)
    print(f"Generated SVAMP dataset with {len(df)} samples")
    return df

def generate_strategyqa_dataset(size=687):
    """
    Generate simulated StrategyQA dataset similar to paper
    """
    print("Generating StrategyQA dataset...")
    
    questions = []
    answers = []
    rationales = []
    
    # Define patterns for commonsense questions
    patterns = [
        # Temporal reasoning
        {"question": "Is the moon visible during the day?", "answer": "Yes", "type": "temporal"},
        # Causal reasoning
        {"question": "If you put ice in a warm room, will it melt?", "answer": "Yes", "type": "causal"},
        # Spatial reasoning
        {"question": "If you turn left at a corner, which direction are you facing?", "answer": "West", "type": "spatial"},
        # Logical reasoning
        {"question": "If all cats are mammals and all mammals have fur, do cats have fur?", "answer": "Yes", "type": "logical"},
        # Counterfactual reasoning
        {"question": "If you didn't water a plant, would it grow?", "answer": "No", "type": "counterfactual"},
    ]
    
    for i in range(size):
        # Randomly select pattern
        pattern = random.choice(patterns)
        
        # Generate rationale based on question type
        if pattern["type"] == "temporal":
            rationale = f"Step 1: The moon reflects sunlight. Step 2: Sunlight is present during the day. Step 3: The moon can be visible when the sun is up. Step 4: Therefore, the answer is Yes"
        elif pattern["type"] == "causal":
            rationale = f"Step 1: Ice is solid water. Step 2: Heat causes water to change from solid to liquid. Step 3: A warm room provides heat. Step 4: Therefore, ice melts in a warm room. The answer is Yes"
        elif pattern["type"] == "spatial":
            rationale = f"Step 1: Directions are relative to your orientation. Step 2: If you're facing north, turning left makes you face west. Step 3: If you're facing south, turning left makes you face east. Step 4: If you're facing east, turning left makes you face north. Step 5: If you're facing west, turning left makes you face south. Step 6: Therefore, the answer is West"
        elif pattern["type"] == "logical":
            rationale = f"Step 1: All cats are mammals. Step 2: All mammals have fur. Step 3: Therefore, cats have fur. The answer is Yes"
        elif pattern["type"] == "counterfactual":
            rationale = f"Step 1: Plants need water to grow. Step 2: Water is necessary for plant growth. Step 3: Without water, plants cannot grow. Step 4: Therefore, if you didn't water a plant, it would not grow. The answer is No"
        
        # Add to dataset
        questions.append(pattern["question"])
        answers.append(pattern["answer"])
        rationales.append(rationale)
    
    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "rationale": rationales
    })
    
    df.to_csv("data/strategyqa.csv", index=False)
    print(f"Generated StrategyQA dataset with {len(df)} samples")
    return df

def main():
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate datasets
    generate_aqua_rat_dataset()
    generate_svamp_dataset()
    generate_strategyqa_dataset()

if __name__ == "__main__":
    main()