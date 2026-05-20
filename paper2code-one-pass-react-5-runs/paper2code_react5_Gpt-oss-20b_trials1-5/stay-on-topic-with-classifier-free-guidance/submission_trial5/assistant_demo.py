"""
assistant_demo.py

Demonstrates assistant‑style prompting (system + user prompt) with CFG.
A simple negative‑prompting example is also shown.

Usage
-----
>>> python assistant_demo.py
"""

from cfg_inference import CfgGenerator, load_gpt2

MODEL_NAME = "gpt2-medium"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond in a friendly tone."
)
USER_PROMPT = "Tell me about the band Halocene."
NEGATIVE_PROMPT = (
    "Please do not mention any real artists or give factual information."
)
GAMMA = 2.0
TEMPERATURE = 0.8
TOP_K = 0
TOP_P = 0.9
MAX_NEW_TOKENS = 80

def main():
    data = load_gpt2(MODEL_NAME)
    model, tokenizer = data["model"], data["tokenizer"]

    # CFG with negative prompting
    gen = CfgGenerator(
        model,
        tokenizer,
        gamma=GAMMA,
        negative_prompt=NEGATIVE_PROMPT,
    )

    combined_prompt = f"{SYSTEM_PROMPT}\nUser: {USER_PROMPT}\nAssistant:"
    output = gen.generate(
        combined_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        return_full_text=True,
    )

    print("\n=== Assistant demo with CFG + negative prompt ===")
    print(output)
    print("\n(End of demo)")

if __name__ == "__main__":
    main()