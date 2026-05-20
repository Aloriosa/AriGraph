#!/bin/bash

# Set up the environment
echo "Setting up the environment..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install torch torchvision torchaudio numpy matplotlib scikit-learn scipy

# Clone the repository (if needed)
echo "Cloning the repository..."
git clone https://github.com/jacksimons15327/snpse_icml.git
cd smpse_icml

# Install the package in development mode
echo "Installing the package in development mode..."
pip install -e .

# Run the reproduction script
echo "Running the reproduction script..."
cd /home/submission

# Execute the main reproduction script
python3 reproduce.py --budget 1000 --rounds 10 --output output.csv

# Verify the output
echo "Verifying the output..."
if [ -f "output.csv" ]; then
    echo "Output file created successfully!"
    echo "Results:"
    cat output.csv
else
    echo "Error: Output file was not created!"
    exit 1
fi

echo "Reproduction completed successfully!"