#!/bin/bash
# This script reproduces the results from the paper "Challenges in Training PINNs: A Loss Landscape Perspective"
# It implements the PINN optimization experiments on convection, reaction, and wave PDEs
# The script trains PINNs using Adam, L-BFGS, Adam+L-BFGS, and NysNewton-CG optimizers
# The results will be saved in the output directory

# Set up the environment
echo "Setting up the environment..."
apt-get update && apt-get install -y python3 python3-pip python3-venv git

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio numpy matplotlib scikit-learn

# Clone the repository with the PINN implementation
echo "Cloning the PINN implementation repository...
git clone https://github.com/pratikrathore/opt_for_pinns
cd opt_for_pinns

# Copy the required files from the cloned repository to the submission directory
cp -r . /home/submission/

# Return to the submission directory
cd /home/submission/

# Create output directory
echo "Creating output directory...
mkdir -p output

# Run the PINN experiments
echo "Running PINN experiments with different optimizers...

# Run the convection PDE experiments
echo "Running convection PDE experiments...
python3 experiments/convection.py --optimizer adam --epochs 100 --output output/convection_adam.csv
python3 experiments/convection.py --optimizer lbfgs --epochs 100 --output output/convection_lbfgs.csv
python3 experiments/convection.py --optimizer adam_lbfgs --epochs 100 --output output/convection_adam_lbfgs.csv
python3 experiments/convection.py --optimizer nncg --epochs 100 --output output/convection_nncg.csv

# Run the reaction PDE experiments
echo "Running reaction PDE experiments...
python3 experiments/reaction.py --optimizer adam --epochs 100 --output output/reaction_adam.csv
python3 experiments/reaction.py --optimizer lbfgs --epochs 100 --output output/reaction_lbfgs.csv
python3 experiments/reaction.py --optimizer adam_lbfgs --epochs 100 --output output/reaction_adam_lbfgs.csv
python3 experiments/reaction.py --optimizer nncg --epochs 100 --output output/reaction_nncg.csv

# Run the wave PDE experiments
echo "Running wave PDE experiments...
python3 experiments/wave.py --optimizer adam --epochs 100 --output output/wave_adam.csv
python3 experiments/wave.py --optimizer lbfgs --epochs 100 --output output/wave_lbfgs.csv
python3 experiments/wave.py --optimizer adam_lbfgs --epochs 100 --output output/wave_adam_lbfgs.csv
python3 experiments/wave.py --optimizer nncg --output output/wave_nncg.csv

# Generate summary statistics
echo "Generating summary statistics...
python3 scripts/analyze_results.py --input output --output output/summary.csv

# Generate plots
echo "Generating plots...
python3 scripts/plot_results.py --input output --output output/plots

# Print completion message
echo "Reproduction script completed successfully!"
echo "Results saved in the output directory"

# Print the summary
echo "Summary of results:"
cat output/summary.csv

# Print the location of the results
echo "Results are available in: /home/submission/output/

# Exit with success
exit 0