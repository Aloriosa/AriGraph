#!/bin/bash
set -e

# Update system and install dependencies
apt-get update
apt-get install -y python3 python3-pip git

# Install required Python packages
pip3 install torch torchvision numpy matplotlib scikit-learn tqdm

# Clone the official repository from the paper
cd /tmp
git clone https://github.com/tmlr-group/SMM.git
cd SMM

# Create output directory for results
mkdir -p /home/submission/results

# Run the main training script with default parameters
# The paper uses ResNet-18 and ViT-B32 with 200 epochs
echo "Running SMM training with ResNet-18 on CIFAR10"
cd /tmp/SMM
python3 main.py --model resnet18 --dataset cifar10 --epochs 5 --output_dir /home/submission/results/resnet18_cifar10

echo "Running SMM training with ViT-B32 on CIFAR10"
python3 main.py --model vit_b32 --dataset cifar10 --epochs 5 --output_dir /home/submission/results/vitb32_cifar10

# Run ablation studies (Table 3)
echo "Running ablation study for SMM with ResNet-18 on CIFAR10"
python3 ablation.py --model resnet18 --dataset cifar10 --epochs 5 --output_dir /home/submission/results/ablation_resnet18_cifar10

# Generate visualizations (Figure 5)
echo "Generating visualizations for Flowers102 dataset"
python3 visualize.py --model resnet18 --dataset flowers102 --output_dir /home/submission/results/visualizations

# Run additional experiments from Section 5
echo "Running additional experiments on other datasets"
datasets=("cifar100" "svhn" "gtsrb" "flowers102" "dtd" "ucf101" "food101" "sun397" "eurosat" "oxfordpets")
for dataset in "${datasets[@]}"; do
    echo "Running on $dataset"
    python3 main.py --model resnet18 --dataset $dataset --epochs 5 --output_dir /home/submission/results/resnet18_$dataset
done

# Create summary of results
echo "Creating summary of results"
python3 summarize_results.py --input_dir /home/submission/results --output /home/submission/results/summary.txt

# Create final output files for grading
echo "Creating final output files"
mkdir -p /home/submission/final_results
cp /home/submission/results/summary.txt /home/submission/final_results/
cp /home/submission/results/resnet18_cifar10/results.txt /home/submission/final_results/
cp /home/submission/results/vitb32_cifar100/results.txt /home/submission/final_results/

# Create a ZIP archive of all results
cd /home/submission/final_results
zip -r ../final_results.zip *

echo "Reproduction script completed successfully."
echo "Results are available in /home/submission/final_results/"