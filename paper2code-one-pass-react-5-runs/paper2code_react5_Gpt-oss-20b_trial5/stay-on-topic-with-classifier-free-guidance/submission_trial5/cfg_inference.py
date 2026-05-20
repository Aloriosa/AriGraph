"""
cfg_inference.py

A minimal implementation of Classifier‑Free Guidance (CFG) for causal language models
according to Eq. 7 of the paper *Stay on topic with Classifier‑Free Guidance*.
The implementation supports:
  * standard CFG (conditioning vs. unconditional)
  * negative‑prompting
  * temperature, top‑k, top‑p sampling
  * generation of arbitrary length (greedy or multinomial)

Only the GPT‑2 family is supported out‑of‑the‑box, but the code is written
generically so that any HuggingFace causal LM can be used.

Usage
-----
>>> from cfg_inference import CfgGenerator
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
>>> gen = CfgGenerator(model, tokenizer, gamma=1.5)
>>> out = gen.generate("The quick brown fox", max_new_tokens=20)
>>> print(out)