#!/usr/bin/env python3
"""
Demo script that shows CFG generation, negative prompting and a simple
chain‑of‑thought style generation.
"""

import os
import json
from cfg import CFGGenerator

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_NAME = "gpt2"  # change to "EleutherAI/gpt-neo-2.7B" or "meta-llama/Llama-2-7b-hf" if GPU is available
OUTPUT_DIR = "outputs"
GAMMA = 1.5
TEMPERATURE = 0.9
TOP_K = 0
TOP_P = 0.95
MAX_TOKENS = 60
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
PROMPTS = [
    "Once upon a time, in a faraway land, there lived a",
    "Translate the following to French: 'Good morning, how are you?'",
    "Explain why the sky is blue in simple terms.",
    # Chain‑of‑thought example
    "What is 23 + 27? First show the addition steps, then the answer.",
]

# Negative prompt used for assistant‑style generation
NEGATIVE_PROMPT = (
    "The prompt below is a question to answer, a task to complete, or a conversation to respond to; "
    "decide which and write an appropriate response."
)

# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------
def main():
    gen = CFGGenerator(model_name=MODEL_NAME)

    results = []

    for idx, prompt in enumerate(PROMPTS, start=1):
        # If the prompt looks like a chain‑of‑thought, we simply generate until an answer token
        if "answer" in prompt.lower():
            output = gen.generate(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                max_new_tokens=MAX_TOKENS,
                gamma=GAMMA,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                seed=SEED,
            )
        else:
            output = gen.generate(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                max_new_tokens=MAX_TOKENS,
                gamma=GAMMA,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                seed=SEED,
            )

        results.append(
            {
                "prompt_id": idx,
                "prompt": prompt,
                "generated": output,
            }
        )
        print(f"--- Prompt {idx} ---")
        print(output)
        print("\n")

    # Save to JSON
    out_path = os.path.join(OUTPUT_DIR, "generated_examples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()