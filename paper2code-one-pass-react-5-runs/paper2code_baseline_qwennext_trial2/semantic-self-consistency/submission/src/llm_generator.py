import requests
import time
import random
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-3.5-turbo"
MAX_RETRIES = 3
TEMPERATURE = 0.7
MAX_TOKENS = 400

class LLMGenerator:
    def __init__(self, api_key: str = API_KEY, model: str = MODEL_NAME):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def generate_response(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        """Generate a single response from the LLM"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "n": 1,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(BASE_URL, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    raise ValueError("No choices returned")
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed after {MAX_RETRIES} attempts: {e}")
                    raise
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
        
        raise RuntimeError("Unexpected state")
    
    def generate_multiple_responses(self, prompt: str, n_responses: int = 10) -> List[str]:
        """Generate multiple responses for a prompt"""
        responses = []
        for _ in tqdm(range(n_responses), desc="Generating responses"):
            try:
                response = self.generate_response(prompt, MAX_TOKENS)
                responses.append(response)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append("")  # Add empty string for error
        return responses

# Example usage
if __name__ == "__main__":
    generator = LLMGenerator()
    prompt = "Solve: 5 + 3 * 2"
    responses = generator.generate_multiple_responses(prompt, 5)
    for i, r in enumerate(responses):
        print(f"Response {i+1}: {r[:100]}...")