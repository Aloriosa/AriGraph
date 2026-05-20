#!/bin/bash
# Set up environment
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate peft bitsandbytes scipy numpy scikit-learn tqdm wandb

# Clone the repository (if needed) and set up the codebase
cd /home/submission

# Create directory structure
mkdir -p models prune trainer scripts eval utils
mkdir -p data/sft

# Download sample dataset for training (Alpaca GPT-4 for Llama)
wget -O data/sft/alpaca_data_gpt4.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

# Copy all source files from submission
cp -r /home/submission/models/* models/
cp -r /home/submission/prune/* prune/
cp -r /home/submission/trainer/* trainer/
cp -r /home/submission/scripts/* scripts/
cp -r /home/submission/eval/* eval/
cp -r /home/submission/utils/* utils/
cp /home/submission/args.py .
cp /home/submission/efficiency_test.py .

# Set environment variables for reproducibility
export PYTHONPATH="/home/submission:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Set seed for reproducibility
export SEED=128

# Run training on Llama-2-7B with APT method (as specified in paper)
echo "Starting APT training on Llama-2-7B with Alpaca GPT-4 dataset..."
python3 run_llama_sft.py \
    --output_dir output/llama2_7b_apt \
    --task_name alpaca_gpt4 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --bf16 True \
    --data_path 'data/sft/alpaca_data_gpt4.json' \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 1000 \
    --log_level info \
    --log_level_replica info \
    --model_max_length 512 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.03 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --tf32 True \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --distillation_type self_momentum \
    --pruner_type running_fisher \
    --param_allocation_strategy running_fisher \
    --mac_constraint 0.7 \
    --pre_tuning_constraint 1.0 \
    --seed $SEED \
    --pruning_start -1 \
    --pruning_stop 5 \
    --num_prunings 16 \
    --pruning_batches 128 \
    --pruning_batch_size 2 \
    --pre_pruning_tuning_steps 200 \
    --sparsity_warmup_epochs 1 \
    --teacher_param_tuning_config dq:0-31,dv:0-31 \
    --student_param_tuning_config dq:0-31,dv:0-31 \
    --warmup_param_tuning_config dq:0-31,dv:0-31 \
    --collect_salience True

# Run evaluation on MMLU benchmark
echo "Running MMLU evaluation..."
python3 scripts/eval/mmlu.sh output/llama2_7b_apt | tee output/llama2_7b_apt/mmlu_results.log

# Run efficiency testing
echo "Running efficiency testing..."
python3 efficiency_test.py --config /home/submission/efficiency_config.json

# Save final results
echo "Saving final results..."
mkdir -p results
cp output/llama2_7b_apt/mmlu_results.log results/
cp output/llama2_7b_apt/trainer_state.json results/

# Print final results
echo "Reproduction complete!"
echo "Results saved in results/ directory"
echo "Expected outcomes:"
echo "- 86.4% task performance on Llama-2-7B with 70% parameter retention"
echo "- 30% memory consumption compared to SOTA pruning methods"
echo "- 2.4x speedup during inference"
echo "- Training memory reduced by 24.2% compared to LoRA"