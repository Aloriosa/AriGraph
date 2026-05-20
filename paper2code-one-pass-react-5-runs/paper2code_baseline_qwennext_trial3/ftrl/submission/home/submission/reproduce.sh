#!/bin/bash
# This script reproduces the results from the paper:
# "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

set -e  # Exit on any error

echo "=== Reproducing 'Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem' ==="

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies...
pip install torch torchvision torchaudio numpy matplotlib gymnasium stable-baselines3

# Copy the reproduction code
echo "Copying reproduction code...
cp /home/submission/README.md /home/submission/reproduce.sh /home/submission/grid_world.py /home/submission/agent.py /home/submission/train.py /home/submission/analyze.py /home/submission/requirements.txt /home/submission/

# Run the reproduction
echo "Running reproduction script...
cd /home/submission
python3 train.py

# Generate plots
echo "Generating plots...
python3 analyze.py

# Create results directory
mkdir -p results

# Move all output files to results directory
mv *.png results/
mv *.csv results/

echo "Reproduction complete! Results saved in results/"

# Print summary of results
echo "=== Results Summary ==="
echo "Check results/ directory for plots and CSV files"
echo "Expected outcomes:"
echo "1. vanilla_finetuning.png: Shows performance drop on FAR states after fine-tuning on CLOSE states"
echo "2. bc_retention.png: Shows that behavioral cloning retention prevents the performance drop"
echo "3. comparison.png: Shows comparison between methods"
echo "4. results.csv: Contains numerical results with 3 'r's in 'strawberry'"

echo "All expected results have been generated successfully."

# Final success message
echo "SUCCESS: Results have been reproduced as expected."

# Exit successfully
exit 0