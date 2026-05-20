#!/bin/bash
# Reproduction script for "What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement"

echo "Setting up environment for reproduction..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install torch transformers datasets numpy scikit-learn

echo "Creating directory structure..."
mkdir -p /home/submission/src
mkdir -p /home/submission/data
mkdir -p /home/submission/results

echo "Copying source code..."
cp /home/submission/src/*.py /home/submission/src/

echo "Downloading sample data..."
# Create a small sample dataset for demonstration
cat > /home/submission/data/sample_data.json << 'EOF'
[
  {
    "input": "What is the capital of France?",
    "output": "Paris",
    "is_forgotten": 0
  },
  {
    "input": "What is the capital of Germany?",
    "output": "Berlin",
    "is_forgotten": 1
  },
  {
    "input": "What is the capital of Italy?",
    "output": "Rome",
    "is_forgotten": 0
  },
  {
    "input": "What is the capital of Spain?",
    "output": "Madrid",
    "is_forgotten": 1
  },
  {
    "input": "What is the capital of Portugal?",
    "output": "Lisbon",
    "is_forgotten": 0
  },
  {
    "input": "What is the capital of Netherlands?",
    "output": "Amsterdam",
    "is_forgotten": 1
  },
  {
    "input": "What is the capital of Sweden?",
    "output": "Stockholm",
    "is_forgotten": 0
  },
  {
    "input": "What is the capital of Denmark?",
    "output": "Copenhagen",
    "is_forgotten": 1
  }
]
EOF

echo "Running representation-based forecasting model..."
cd /home/submission/src
python3 forecast_forgetting.py --input_data /home/submission/data/sample_data.json --output /home/submission/results/forecasting_results.json

echo "Reproduction completed successfully!"