# FOA: Forward Optimization Adaptation for Test-Time Adaptation of Quantized Vision Transformers

This repository reproduces the FOA (Forward Optimization Adaptation) method from the paper "Forward Optimization Adaptation for Test-Time Adaptation of Quantized Vision Transformers". FOA is a gradient-free test-time adaptation method that optimizes learnable prompts using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to improve out-of-distribution generalization of quantized ViT models.

## Key Features

- **Gradient-Free Adaptation**: Uses CMA-ES for prompt optimization without backpropagation
- **Quantization Compatible**: Works with 8-bit and 6-bit quantized ViT models
- **Minimal Memory Overhead**: Only adapts prompt embeddings (3MB overhead vs 5165MB for TENT)
- **Activation Shifting**: Aligns test-time activations with source domain statistics
- **Unsupervised Fitness Function**: Combines entropy minimization and activation discrepancy

## Reproduction Results

Running `reproduce.sh` will:
1. Load a pre-trained ViT-B/16 model
2. Apply PTQ4ViT quantization to 8-bit precision
3. Adapt the model using FOA on ImageNet-C (Level 5)
4. Evaluate accuracy and Expected Calibration Error (ECE)

Expected outcomes (matching paper):
- **Accuracy**: 66.3% on ImageNet-C (vs 59.6% for TENT)
- **ECE**: 3.2% on ImageNet-C (vs 18.5% for TENT)
- **Memory Reduction**: 24x less memory than TENT

## Dependencies

- PyTorch
- torchvision
- timm
- cma
- numpy
- tqdm

## Usage

```bash
# Clone this repository
git clone https://github.com/your-repo/foa-reproduction.git
cd foa-reproduction

# Run reproduction script
bash reproduce.sh
```

The results will be saved in `results/` directory as CSV files.