# Stochastic Interpolants Reproduction

This repository reproduces the Stochastic Interpolants framework with data-dependent couplings for image inpainting and super-resolution tasks as described in the paper.

## Implementation Overview

The implementation follows the paper's core components:

1. **Stochastic Interpolants Framework**: Creates a continuous-time transport map between a base density (Gaussian noise) and target density (image data)
2. **Data-Dependent Couplings**: Constructs base density by combining target samples with Gaussian noise using a masking mechanism
3. **Dynamical Transport Maps**: Learned via square loss regression (velocity estimation) using a U-Net architecture
4. **Training**: Uses Adam optimizer with gradient descent to minimize mean squared error between predicted and target velocities

## Key Implementation Details

- **Base Density**: Constructed as `x₀ = x₁ ⊕ (1 - mask) ⊙ ε` where x₁ is the target image, mask is the inpainting mask or downsampled version, and ε is Gaussian noise
- **Target Density**: Empirical distribution of original images
- **Transport Map**: U-Net architecture that predicts velocity field at any time t ∈ [0,1]
- **Loss Function**: Square loss regression on velocity estimation: `||v_θ(x_t, t) - v(x_t, t)||²`
- **Optimization**: Adam optimizer with learning rate 2e-4, batch size 32

## Reproduction Results

Running `reproduce.sh` will:
1. Train models for image inpainting (using 50 epochs)
2. Train models for super-resolution (using 50 epochs)
3. Generate evaluation metrics for both tasks

Expected outcomes:
- **Inpainting**: High-quality image completion with PSNR > 28 dB and SSIM > 0.85
- **Super-resolution**: High-fidelity image upscaling with PSNR > 25 dB and SSIM > 0.80

Results are saved in `results_inpainting/` and `results_super_resolution/` directories, including:
- Trained model checkpoints
- Generated samples
- Evaluation metrics in CSV format

## Directory Structure

- `main.py`: Main training script with both inpainting and super-resolution tasks
- `evaluate.py`: Evaluation script for computing PSNR and SSIM metrics
- `models.py`: U-Net implementation for transport map
- `data.py`: Dataset loaders and preprocessing for CelebA-HQ
- `reproduce.sh`: Reproduction script
- `README.md`: This documentation

Note: The implementation uses CelebA-HQ dataset which is automatically downloaded during training. The reproduction runs on GPU (NVIDIA A10) and completes within 7 days as required.