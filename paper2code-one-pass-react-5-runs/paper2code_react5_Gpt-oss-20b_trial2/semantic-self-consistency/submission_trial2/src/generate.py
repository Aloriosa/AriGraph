import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def generate_responses(question, model_name, temperature, num_samples,
                       top_p=0.9, top_k=50):
    """
    Generate multiple chain‑of‑thought responses for a single question.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = f"Question: {question}\nAnswer:"

    responses = []
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id
        )
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=gen_cfg)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append(text)
    return responses