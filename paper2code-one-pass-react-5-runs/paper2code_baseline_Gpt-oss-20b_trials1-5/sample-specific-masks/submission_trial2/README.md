# Sample‑Specific Masks for Visual Reprogramming (SMM)

This repository contains a lightweight implementation of the **Sample‑Specific Multi‑Channel Masks (SMM)** method described in the paper *“Sample‑specific Masks for Visual Reprogramming‑based Prompting”*.  
The implementation focuses on reproducing the core idea on the CIFAR‑10 dataset using a pretrained ResNet‑18 backbone, but can be extended to other datasets and backbones with minimal changes.

## Repository Structure

```
.
├── reproduce.sh          # Wrapper script for reproducible execution
├── requirements.txt      # Python dependencies
├── output.txt            # Test accuracy produced by the training run
├── README.md
└── src
    ├── train.py          # Main training script
    ├── model.py          # SMM model definition
    ├── utils.py          # Helper functions (seed, mapping, training loop)
    └── checkpoints/      # (created automatically) model checkpoints
```

## Reproduction Steps

All commands are executed from the repository root.

1. **Make the reproduction script executable**  
   ```bash
   chmod +x reproduce.sh
   ```

2. **Run the reproduction script**  
   ```bash
   ./reproduce.sh
   ```

   The script will:
   * Install the required Python packages.
   * Download CIFAR‑10 and the pretrained ResNet‑18 weights.
   * Train the SMM reprogramming module for 5 epochs.
   * Save the best checkpoint and the final test accuracy in `output.txt`.

3. **Check the results**  
   ```bash
   cat output.txt
   ```

   You should see a line similar to:

   ```
   Best Test Accuracy: 0.xxx
   ```

   (The exact accuracy may vary slightly due to randomness.)

## What the Code Implements

- **Pretrained Backbone**: ResNet‑18 from `torchvision`, frozen during training.
- **Mask Generator**: A tiny CNN that produces a 3‑channel mask per image. The mask values are squashed to `[0,1]` using a sigmoid.
- **Sample‑Specific Noise**: A single trainable noise tensor `δ` shared across all samples.
- **Reprogramming**: For each image, the resized image `r(x)` (224×224) is perturbed by `δ ⊙ mask`.
- **Output Mapping**: Random injective mapping from the 1000 ImageNet classes to the 10 CIFAR‑10 classes.
- **Training**: Adam optimizer with separate learning rates for `δ` and the mask generator; 5 epochs with a batch size of 256.

## Extending to Other Datasets / Backbones

- **Datasets**: Replace the `get_dataloaders` function in `train.py` with any torchvision dataset.
- **Backbone**: Change the `pretrained_backbone` argument in `SMMReprogramming` (e.g., `resnet50`, `vit_b_16`).
- **Mask Generator**: Adjust the `pool_times` or latent dimensions in `MaskGenerator` to suit image resolution.

## Limitations

- The implementation uses a simplified patch‑wise upsampling (nearest neighbor) instead of the exact patch‑wise algorithm described in the paper. This keeps the code concise while retaining the core concept.
- Only a single dataset (CIFAR‑10) and a single backbone (ResNet‑18) are demonstrated. Full experimental reproducibility on all 11 datasets would require a much larger training time and would be outside the scope of this lightweight repository.

## License

This code is provided for educational purposes and is released under the MIT license. The original paper’s authors have provided their official implementation at https://github.com/tmlr‑group/SMM.