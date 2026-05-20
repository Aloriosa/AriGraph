import pickle
import random
import numpy as np

# Generate synthetic pairwise preference dataset as described in paper_card_0004
# This simulates a 200MB dataset with prompt-response pairs filtered by toxicity threshold

def generate_synthetic_pairwise_dataset(num_pairs=10000):
    """
    Generate a synthetic pairwise preference dataset for DPO training.
    Each pair consists of:
    - a prompt
    - a preferred response (less toxic)
    - a dispreferred response (more toxic)
    """
    
    # Define prompt templates from paper_card_0005
    prompt_templates = [
        "generate a response to: {}",
        "write a response to: {}",
        "respond to: {}",
        "answer: {}"
    ]
    
    # Define some common toxic and non-toxic phrases
    toxic_phrases = [
        "sh*t", "f*ck", "asshole", "b*tch", "d*ck", "c*nt", "p*ssy", "n*gger",
        "you're stupid", "everyone hates you", "kill yourself", "you're ugly",
        "nobody likes you", "you deserve to die", "f*ck you", "s*ck my d*ck",
        "i hate you", "you're a waste of space", "go die", "you're worthless"
    ]
    
    non_toxic_phrases = [
        "I'm sorry to hear that.", "That sounds challenging.", "I understand your perspective.",
        "Thank you for sharing.", "I appreciate your honesty.", "Let me think about that.",
        "That's an interesting point.", "I see what you mean.", "Thanks for your input.",
        "I'm here to help.", "How can I assist you?", "I'd be happy to help.",
        "That's a valid concern.", "I understand your feelings.", "I appreciate your feedback."
    ]
    
    # Generate prompts
    prompts = []
    for i in range(num_pairs):
        # Create a prompt with a mix of topics
        topic = random.choice([
            "climate change", "politics", "technology", "health", "education",
            "relationships", "work", "sports", "entertainment", "food"
        ])
        prompt_text = f"Tell me about {topic}."
        prompts.append(prompt_text)
    
    # Generate pairwise comparisons
    pairs = []
    for i in range(num_pairs):
        prompt = prompts[i]
        
        # Select a template
        template = random.choice(prompt_templates)
        formatted_prompt = template.format(prompt)
        
        # Select toxic and non-toxic responses
        toxic_response = random.choice(toxic_phrases)
        non_toxic_response = random.choice(non_toxic_phrases)
        
        # Ensure the responses are different
        if toxic_response == non_toxic_response:
            non_toxic_response = random.choice([p for p in non_toxic_phrases if p != toxic_response])
        
        # Create the pair
        pair = {
            "prompt": formatted_prompt,
            "chosen": non_toxic_response,  # preferred response (less toxic)
            "rejected": toxic_response,    # dispreferred response (more toxic)
            "prompt_id": i
        }
        
        pairs.append(pair)
    
    # Save dataset
    with open("data/pairwise_dataset.pkl", "wb") as f:
        pickle.dump(pairs, f)
    
    print(f"Generated {len(pairs)} pairwise preference pairs")
    print(f"Dataset saved to data/pairwise_dataset.pkl")

if __name__ == "__main__":
    generate_synthetic_pairwise_dataset()