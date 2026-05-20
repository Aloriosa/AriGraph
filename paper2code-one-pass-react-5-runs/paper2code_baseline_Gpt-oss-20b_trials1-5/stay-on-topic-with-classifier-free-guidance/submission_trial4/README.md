# Classifier‑Free Guidance (CFG) Reproduction

This repository reproduces the core idea of the paper *“Stay on topic with Classifier‑Free Guidance”* by demonstrating how CFG can be applied to an autoregressive language model (GPT‑2) to increase prompt adherence.

## What we implemented

* **CFG inference** – For every token, we compute the conditioned logits (model + prompt) and the unconditioned logits (model + generated tokens only).  
  We then re‑weight the log‑probabilities as

  ```text
  logp_cfg = logp_unc + γ * (logp_cond – logp_unc)
  ```

  where `γ` is the guidance strength (default `γ = 1.5`).

* **Sampling** – Temperature scaling and nucleus (`top_p`) sampling are applied to the re‑weighted distribution.

* **Evaluation** – We run CFG on two simple prompts and print the generated text.  
  Additionally, we count how many times a keyword from the prompt appears in the output as a crude metric of prompt adherence.

## Reproduction steps

1. **Clone the repository** (or copy the files into `/home/submission/`).

2. **Run the reproducibility script**

   ```bash
   bash reproduce.sh
   ```

   The script installs Python, PyTorch, and Hugging Face Transformers, then executes the generation script.

3. **Check the results**

   The script writes the generated texts and keyword counts to `results.txt`.

   ```text
   $ cat results.txt
   Prompt: Translate to French: Hello world.
   Output: Hello world. Bonjour le monde.
   Keyword 'world' appears 1 times in output.

   Prompt: Write a short story about a dragon.
   Output: Once upon a time, a dragon lived in the mountain...
   Keyword 'dragon' appears 1 times in output.
   ```

   The outputs show that the model is more likely to keep the prompt content when guided with CFG (`γ = 1.5`).

## What you should see

Running the script on a fresh Ubuntu 24.04 container with an NVIDIA A10 GPU will produce:

* Two generated passages (one for each prompt).
* A keyword‑appearance statistic for each passage.
* A final message indicating that reproduction is complete.

The entire repository contains only source code and small text files – well under the 1 GB limit. No large model weights or trained checkpoints are committed.

## Extending the experiment

* Try different `γ` values (e.g., `1.0`, `1.75`, `2.0`) by editing the `generate_with_cfg.py` script.
* Replace the prompts with your own.
* Use a larger model (e.g., `gpt2-medium`) by changing `model_name` in the script.

Enjoy experimenting with Classifier‑Free Guidance!