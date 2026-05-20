#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y python3 python3-pip git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install stable-baselines3[extra] gymnasium[box2d] numpy matplotlib scikit-learn

# Create directory structure
mkdir -p /home/submission/rice
cd /home/submission/rice

# Copy all source files
cp /home/submission/*.py .

# Download and extract Mujoco environments (if needed)
# Note: We'll use Gymnasium's Mujoco environments which are available via pip

# Run the RICE implementation
python3 train_rice.py --env HalfCheetah-v4 --pretrained_model ./pretrained_halfcheetah.zip --output_dir ./results --seed 42 --total_timesteps 1000000

# Generate evaluation results
python3 evaluate_rice.py --env HalfCheetah-v4 --model_path ./results/final_model.zip --episodes 100 --output ./results/evaluation_results.csv

# Create final output directory for grading
mkdir -p /home/submission/output
cp ./results/evaluation_results.csv /home/submission/output/
cp ./results/training_log.csv /home/submission/output/

echo "RICE reproduction completed. Results saved to /home/submission/output/"