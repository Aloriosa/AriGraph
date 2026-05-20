#!/bin/bash

# Set up environment
echo "Setting up environment..."
apt-get update && apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy scipy matplotlib torch scikit-learn

# Clone the repository (if needed)
echo "Cloning repository..."
git clone https://github.com/jacksimons15327/snpse_icml /tmp/snpse_icml
cp -r /tmp/snpse_icml/* .

# Run the reproduction script
echo "Running reproduction script..."
python3 reproduce.py

echo "Reproduction complete!"