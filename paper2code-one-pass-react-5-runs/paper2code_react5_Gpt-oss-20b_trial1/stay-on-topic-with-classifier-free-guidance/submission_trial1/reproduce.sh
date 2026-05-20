#!/usr/bin/env bash
set -euo pipefail

# 1. Install system packages
apt-get update -y
apt-get install -y python3 python3-pip git

# 2. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create output directory
mkdir -p output

# 4. Run CFG demo and benchmarks
python3 scripts/generate_cfg.py \
    --model_name gpt2-medium \
    --output_dir output \
    --max_new_tokens 60 \
    --temperature 1.0 \
    --top_k 0 \
    --top_p 0.95 \
    --seed 1234 \
    --evaluate_lambada \
    --evaluate_perplexity \
    --evaluate_entropy \
    --evaluate_cot \
    --evaluate_self_consistency

# 5. Run assistant demo (negative prompting)
python3 scripts/assistant_demo.py \
    --model_name gpt2-medium \
    --output_file output/assistant_demo.txt \
    --negative_prompt "The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response." \
    --system_prompt "The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write a sad response." \
    --user_prompt "What is the capital of France?" \
    --gamma 3.0 \
    --max_new_tokens 60

echo "Reproduction finished. Generated files are in the 'output' directory."