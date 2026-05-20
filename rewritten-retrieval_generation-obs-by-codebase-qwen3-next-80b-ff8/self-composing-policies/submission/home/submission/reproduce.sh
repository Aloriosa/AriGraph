#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip git curl

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium[box2d] dm-control metaworld numpy matplotlib tensorboard pandas tqdm scikit-learn

# Clone the repository if not already present (for completeness)
cd /home/submission

# Create necessary directories
mkdir -p data runs_all logs

# Download and extract Meta-World dataset (if needed)
# Note: Meta-World will be downloaded automatically by the library on first use

# Run the main training script for CompoNet on Meta-World
python3 main.py --model_type componet --seed 1 --total_timesteps 1000000 --task_sequence meta_world_20 --save_dir runs_all/componet_seed1 --track False

# Run evaluation and generate results
python3 process_results.py --runs-dir runs_all --save-csv data/agg_results.csv --eval-csv data/eval_results.csv

# Generate plots (if required)
python3 plot_results.py

echo "Reproduction complete. Results saved in data/agg_results.csv and runs_all/"