#!/bin/bash

# Set up the environment for reproduction
echo "Setting up environment for reproduction of 'Stay on topic with Classifier-Free Guidance'"

# Update package list and install system dependencies
echo "Updating package list..."
apt-get update

echo "Installing system dependencies..."
apt-get install -y python3 python3-pip python3-venv git curl unzip

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies...
pip install torch transformers datasets evaluate scikit-learn numpy pandas matplotlib seaborn

# Create directories for results and plots
echo "Creating result directories..."
mkdir -p results plots

# Download and run experiments
echo "Running experiments..."

# 1. LAMBADA with LLaMA-7B
echo "Running LAMBADA experiment with LLaMA-7B..."
python3 experiments/lambada.py --model_name "meta-llama/Llama-7B" --output "results/lambada_results.csv" --gamma_values 1.0 1.25 1.5 1.75 2.0 3.0

# 2. GSM8K with WizardLM-30B
echo "Running GSM8K experiment with WizardLM-30B...
python3 experiments/gsm8k.py --model_name "WizardLM/WizardLM-30B" --output "results/gsm8k_results.csv" --gamma_values 1.0 1.25 1.5 1.75 2.0 3.0

# 3. HumanEval with CodeGen-6B
echo "Running HumanEval experiment with CodeGen-6B...
python3 experiments/humaneval.py --model_name "NVIDIA/CodeGen-6B" --output "results/humaneval_results.csv" --gamma_values 1.0 1.25 1.5 1.75 2.0 3.0

# 4. Assistant system prompts
echo "Running assistant system prompts experiment...
python3 experiments/assistant.py --output "results/system_prompt_results.csv" --gamma_values 1.0 1.25 1.5 1.75 2.0 3.0

# Generate plots
echo "Generating plots...
python3 plots/generate_plots.py

echo "Reproduction complete! Results saved in 'results/' directory"