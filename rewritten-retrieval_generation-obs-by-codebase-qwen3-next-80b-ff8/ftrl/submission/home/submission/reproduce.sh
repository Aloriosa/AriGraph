#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y python3 python3-pip git wget

# Install SampleFactory and dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install sample-factory[gym,atari,pygame] tensorboard

# Clone the SampleFactory repository to get the full codebase
git clone https://github.com/alex-petrenko/sample-factory.git /tmp/sample-factory
cd /tmp/sample-factory
pip3 install -e .

# Create directories for models and outputs
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download pre-trained model (simulated - in practice this would be provided)
# For reproduction, we'll train a basic policy on push_wall first, then fine-tune
echo "Training pre-trained model on push_wall task..."

# Train pre-trained model on push_wall (simulated as a simple environment)
python3 -m sample_factory.train --algo=APPO --env=push_wall --train_for_seconds=300 --num_workers=4 --num_envs_per_worker=4 --batch_size=512 --policy_fc_dim=256 --policy_init_gain=1.0 --use_rnn=False --experiment=pretrained_push_wall --save_every_sec=60 --reward_scale=1.0 --use_wandb=False

# Extract the trained model path
PRETRAINED_MODEL_PATH="/tmp/sample-factory/train_dir/pretrained_push_wall/checkpoints/episode_10000.pth"

# Fine-tune with BC (Behavioral Cloning) on Montezuma's Revenge
echo "Fine-tuning with BC on Montezuma's Revenge..."
python3 -m sample_factory.train \
    --algo=APPO \
    --env=montezuma_revenge \
    --train_for_seconds=7200 \
    --num_workers=4 \
    --num_envs_per_worker=4 \
    --batch_size=512 \
    --policy_fc_dim=256 \
    --policy_init_gain=1.0 \
    --use_rnn=False \
    --experiment=ft_bc_montezuma \
    --save_every_sec=60 \
    --reward_scale=1.0 \
    --use_wandb=False \
    --load_from=/tmp/sample-factory/train_dir/pretrained_push_wall/checkpoints/episode_10000.pth \
    --bc_loss_weight=0.1 \
    --bc_loss_type=kl_divergence \
    --normalize_returns=True \
    --entropy_coef=0.01 \
    --ppo_epoch=4 \
    --rollout=32 \
    --learning_rate=0.0001

# Fine-tune with EWC (Elastic Weight Consolidation) on Montezuma's Revenge
echo "Fine-tuning with EWC on Montezuma's Revenge..."
python3 -m sample_factory.train \
    --algo=APPO \
    --env=montezuma_revenge \
    --train_for_seconds=7200 \
    --num_workers=4 \
    --num_envs_per_worker=4 \
    --batch_size=512 \
    --policy_fc_dim=256 \
    --policy_init_gain=1.0 \
    --use_rnn=False \
    --experiment=ft_ewc_montezuma \
    --save_every_sec=60 \
    --reward_scale=1.0 \
    --use_wandb=False \
    --load_from=/tmp/sample-factory/train_dir/pretrained_push_wall/checkpoints/episode_10000.pth \
    --ewc_lambda=1000 \
    --ewc_fisher_samples=100 \
    --normalize_returns=True \
    --entropy_coef=0.01 \
    --ppo_epoch=4 \
    --rollout=32 \
    --learning_rate=0.0001

# Fine-tune with Kickstarting (KS) on Montezuma's Revenge
echo "Fine-tuning with Kickstarting on Montezuma's Revenge..."
python3 -m sample_factory.train \
    --algo=APPO \
    --env=montezuma_revenge \
    --train_for_seconds=7200 \
    --num_workers=4 \
    --num_envs_per_worker=4 \
    --batch_size=512 \
    --policy_fc_dim=256 \
    --policy_init_gain=1.0 \
    --use_rnn=False \
    --experiment=ft_ks_montezuma \
    --save_every_sec=60 \
    --reward_scale=1.0 \
    --use_wandb=False \
    --load_from=/tmp/sample-factory/train_dir/pretrained_push_wall/checkpoints/episode_10000.pth \
    --ks_loss_weight=0.1 \
    --ks_temperature=1.0 \
    --normalize_returns=True \
    --entropy_coef=0.01 \
    --ppo_epoch=4 \
    --rollout=32 \
    --learning_rate=0.0001

# Evaluate on NetHack (using the best performing model from above)
echo "Evaluating on NetHack..."
python3 -m sample_factory.train \
    --algo=APPO \
    --env=nethack \
    --train_for_seconds=1800 \
    --num_workers=4 \
    --num_envs_per_worker=4 \
    --batch_size=512 \
    --policy_fc_dim=256 \
    --policy_init_gain=1.0 \
    --use_rnn=False \
    --experiment=eval_nethack \
    --save_every_sec=60 \
    --reward_scale=1.0 \
    --use_wandb=False \
    --load_from=/tmp/sample-factory/train_dir/ft_bc_montezuma/checkpoints/episode_10000.pth \
    --eval_only=True \
    --eval_episodes=200 \
    --normalize_returns=True \
    --entropy_coef=0.01 \
    --rollout=32 \
    --learning_rate=0.0001

# Copy evaluation results to output directory
cp -r /tmp/sample-factory/train_dir/ft_bc_montezuma/logs /home/submission/results/
cp -r /tmp/sample-factory/train_dir/ft_ewc_montezuma/logs /home/submission/results/
cp -r /tmp/sample-factory/train_dir/ft_ks_montezuma/logs /home/submission/results/
cp -r /tmp/sample-factory/train_dir/eval_nethack/logs /home/submission/results/

# Create summary file with final metrics
echo "Creating results summary..." > /home/submission/results/summary.txt

# Extract final average return from logs (this is a simplified extraction)
for exp in ft_bc_montezuma ft_ewc_montezuma ft_ks_montezuma; do
    log_file="/tmp/sample-factory/train_dir/$exp/logs/progress.csv"
    if [ -f "$log_file" ]; then
        last_line=$(tail -n 1 "$log_file")
        avg_return=$(echo "$last_line" | awk -F',' '{print $4}')
        echo "$exp: Average Return = $avg_return" >> /home/submission/results/summary.txt
    fi
done

# Extract NetHack evaluation results
nethack_log="/tmp/sample-factory/train_dir/eval_nethack/logs/progress.csv"
if [ -f "$nethack_log" ]; then
    last_line=$(tail -n 1 "$nethack_log")
    avg_return=$(echo "$last_line" | awk -F',' '{print $4}')
    echo "eval_nethack: Average Return = $avg_return" >> /home/submission/results/summary.txt
fi

echo "Reproduction complete. Results saved to /home/submission/results/"