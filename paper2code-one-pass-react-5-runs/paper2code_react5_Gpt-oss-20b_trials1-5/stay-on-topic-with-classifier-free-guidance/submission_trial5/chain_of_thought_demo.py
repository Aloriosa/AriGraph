"""
chain_of_thought_demo.py

Demonstrates chain‑of‑thought (CoT) prompting with CFG on a simple arithmetic
question using GPT‑2‑medium.

Usage
-----
>>> python chain_of_thought_demo.py
"""

from cfg_inference import CfgGenerator, load_gpt2

MODEL_NAME = "gpt2-medium"
GAMMA = 1.5
TEMPERATURE = 0.7
TOP_K = 0
TOP_P = 0.9
MAX_NEW_TOKENS = 60

prompt = (
    "Solve the following math problem step by step:\n"
    "What is 23 + 19? "
)

def main():
    data = load_gpt2(MODEL_NAME)
    model, tokenizer = data["model"], data["tokenizer"]

    gen = CfgGenerator(model, tokenizer, gamma=GAMMA)

    output = gen.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        return_full_text=True,
    )

    print("\n=== Chain‑of‑Thought with CFG ===")
    print(output)
    print("\n(End of demo)")

if __name__ == "__main__":
    main()