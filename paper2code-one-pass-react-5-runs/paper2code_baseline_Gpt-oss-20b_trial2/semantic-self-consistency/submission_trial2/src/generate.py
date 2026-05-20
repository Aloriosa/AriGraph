"""
Generation utilities: produce multiple CoT samples for a given question.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List

from .config import (
    NUM_SAMPLES,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
    TOP_P,
    REPEAT,
    DEVICE,
)

# Cache the model to avoid re‑loading for every call
GEN_MODEL = None
GEN_TOKENIZER = None


def _load_gen_model():
    global GEN_MODEL, GEN_TOKENIZER
    if GEN_MODEL is None:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        GEN_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        GEN_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    return GEN_TOKENIZER, GEN_MODEL


def build_prompt(shots: List[tuple], question: str) -> str:
    """Create the 8‑shot chain‑of‑thought prompt."""
    lines = []
    for q, a in shots:
        lines.append(f"Q: {q}\nA: {a}\n")
    lines.append(f"Q: {question}\nA:")
    return "\n".join(lines)


def generate_responses(
    question: str,
    shots: List[tuple],
    num_samples: int = NUM_SAMPLES,
) -> List[str]:
    """Generate multiple CoT responses for a question."""
    tokenizer, model = _load_gen_model()
    prompt = build_prompt(shots, question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    ).to(DEVICE)

    all_responses = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        # Generation
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            max_new_tokens=MAX_TOKENS,
            repetition_penalty=1.2 if REPEAT else 1.0,
        )
        output_ids = model.generate(**gen_kwargs)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # The output contains the prompt + answer; strip the prompt
        answer = output_text[len(prompt) :]
        all_responses.append(answer.strip())
    return all_responses