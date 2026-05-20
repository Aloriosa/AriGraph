"""
Simple prompt helpers for each dataset.
The prompts are intentionally lightweight – they can be extended
to match exactly the wording used in the paper.
"""

def gsm8k_prompt(question, examples=None):
    prompt = ""
    if examples:
        for q, a in examples:
            prompt += f"Question: {q}\nAnswer: {a}\n"
    prompt += f"Question: {question}\nAnswer: Let's think step by step."
    return prompt

def strategyqa_prompt(question, examples=None):
    prompt = ""
    if examples:
        for q, a in examples:
            prompt += f"Q: {q}\nA: {a}\n"
    prompt += f"Q: {question}\nA: Let's think step by step."
    return prompt

def truthfulqa_prompt(question, examples=None):
    prompt = ""
    if examples:
        for q, a in examples:
            prompt += f"Q: {q}\nA: {a}\n"
    prompt += f"Q: {question}\nA: "
    return prompt

def scienceqa_prompt(question, examples=None):
    prompt = ""
    if examples:
        for q, a in examples:
            prompt += f"Question: {q}\nAnswer: {a}\n"
    prompt += f"Question: {question}\nAnswer: "
    return prompt