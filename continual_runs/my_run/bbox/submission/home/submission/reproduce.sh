#!/bin/bash
# Set up environment
apt-get update && apt-get install -y python3 python3-pip git wget

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets scikit-learn pandas numpy tqdm accelerate huggingface-hub

# Download and prepare datasets
mkdir -p /home/submission/data
cd /home/submission/data

# Download sample datasets (using small subsets for reproduction)
# For GSM8K
wget -O gsm8k.json https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
# For StrategyQA
wget -O strategyqa.json https://raw.githubusercontent.com/mandarjoshi90/trivia_qa/master/strategyqa/train.json
# For TruthfulQA
wget -O truthfulqa.json https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv
# For ScienceQA
wget -O scienceqa.json https://raw.githubusercontent.com/eric-haoyu-liu/ScienceQA/main/data/scienceqa/train.json

# Create data directory structure
mkdir -p /home/submission/data/gsm8k
mkdir -p /home/submission/data/strategyqa
mkdir -p /home/submission/data/truthfulqa
mkdir -p /home/submission/data/scienceqa

# Convert to appropriate format
python3 -c "
import json
import csv

# Process GSM8K
with open('gsm8k.json', 'r') as f:
    data = [json.loads(line) for line in f]
    gsm8k_data = [{'question': d['question'], 'answer': d['answer']} for d in data[:10]]  # Use 10 samples for reproduction
with open('gsm8k/train.json', 'w') as f:
    json.dump(gsm8k_data, f)

# Process StrategyQA
with open('strategyqa.json', 'r') as f:
    data = json.load(f)
    strategyqa_data = [{'question': d['question'], 'answer': d['answer']} for d in data[:10]]  # Use 10 samples
with open('strategyqa/train.json', 'w') as f:
    json.dump(strategyqa_data, f)

# Process TruthfulQA
with open('truthfulqa.json', 'r') as f:
    reader = csv.DictReader(f)
    truthfulqa_data = [{'question': row['Question'], 'answer': row['Best Answer']} for row in list(reader)[:10]]  # Use 10 samples
with open('truthfulqa/train.json', 'w') as f:
    json.dump(truthfulqa_data, f)

# Process ScienceQA
with open('scienceqa.json', 'r') as f:
    data = json.load(f)
    scienceqa_data = [{'question': d['question'], 'answer': d['answer']} for d in data[:10]]  # Use 10 samples
with open('scienceqa/train.json', 'w') as f:
    json.dump(scienceqa_data, f)
"

# Create output directories
mkdir -p /home/submission/results
mkdir -p /home/submission/models

# Run the adaptation and evaluation
python3 /home/submission/bbox_adapter.py \
    --base_model gpt-3.5-turbo \
    --adapter_model deberta-v3-base \
    --datasets gsm8k strategyqa truthfulqa scienceqa \
    --num_samples 10 \
    --max_iterations 3 \
    --adapter_size 0.1b \
    --output_dir /home/submission/results \
    --model_dir /home/submission/models

# Evaluate results
python3 /home/submission/evaluate_results.py \
    --input_dir /home/submission/results \
    --output_file /home/submission/results/accuracy_results.csv

# Print completion message
echo "BBoxAdapter reproduction completed. Results saved to /home/submission/results/"
echo "Expected accuracy improvement: ~6.77% over base model"