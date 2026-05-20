import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_wikitext_train():
    """
    Load the raw Wikitext-2 training split.
    """
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def generate_continuation(
    tokenizer,
    model,
    prompt,
    max_length=50,
    do_sample=False,
    temperature=1.0,
    top_k=50,
):
    """
    Generate a continuation given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    # Ensure we don't exceed model max length
    if input_ids.size(1) >= max_length:
        input_ids = input_ids[:, -max_length + 1 :]
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Remove the prompt tokens
    continuation_ids = generated_ids[0, input_ids.size(1) :]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True)

def build_pairwise_dataset(
    tokenizer,
    base_model,
    num_pairs=200,
    seed=42,
    max_length=50,
    temperature=1.2,
    top_k=50,
):
    """
    Build a synthetic pairwise dataset from Wikitext-2.
    For each prompt, the chosen continuation is greedy, the rejected is sampled.
    """
    random.seed(seed)
    wikitext = load_wikitext_train()
    prompts = [line["text"].strip() for line in wikitext if line["text"].strip()]
    random.shuffle(prompts)

    pairs = []
    base_model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts[:num_pairs], desc="Generating pairs"):
            chosen = generate_continuation(
                tokenizer, base_model, prompt, max_length, do_sample=False
            )
            rejected = generate_continuation(
                tokenizer,
                base_model,
                prompt,
                max_length,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
            )
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    df = pd.DataFrame(pairs)
    return load_dataset("pandas", data_files={"train": df})["train"]