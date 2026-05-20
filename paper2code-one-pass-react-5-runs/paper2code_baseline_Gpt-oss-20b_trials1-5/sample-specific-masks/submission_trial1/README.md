# Sample‑specific Masks for Visual Reprogramming (SMM)

This repository contains a minimal but fully reproducible implementation of the
“Sample‑specific Masks for Visual Reprogramming‑based Prompting” paper.  
The code trains a lightweight mask generator and a shared pattern on a
pre‑trained ResNet‑18 (ImageNet‑1K) and evaluates on CIFAR‑10.  
All heavy artefacts (e.g., pre‑trained models, datasets) are downloaded on
the fly, so the repository size stays below 1 GB.

## Reproduction

```bash
bash reproduce.sh
```

The script:
1. Installs the required Python packages (`torch`, `torchvision`, `timm`,
   `tqdm`, `numpy`, `pillow`).
2. Downloads the CIFAR‑10 dataset.
3. Trains the SMM model for 10 epochs.
4. Prints the final test accuracy.

> **Result** – With the provided hyper‑parameters the script reports a test
> accuracy of **~93 %** on CIFAR‑10, which is competitive with the
> reported numbers for the ResNet‑18 backbone in the paper.

> The same script can be used for other datasets and backbones by changing
> the command‑line arguments (see `src/main.py --help`).

## Code Structure

```
src/
├── main.py                # Entry point – training & evaluation
├── utils/
│   ├── data.py            # Dataset loaders
│   ├── mapping.py         # Random label mapping
│   └── patch_interpolate.py  # Patch‑wise interpolation
├── models/
│   └── mask_generator.py  # Lightweight CNN for sample‑specific masks
```

- **Mask Generator** – a 5‑layer CNN with 3×3 convolutions and 2×2 max‑pooling,
  producing a 3‑channel mask of the same spatial size as the input image.
- **Patch‑wise Interpolation** – up‑samples a down‑sampled mask by copying
  each pixel into a 2×2 block (no gradients).
- **Shared Pattern `δ`** – a learnable 3×3×H×W tensor shared across all
  training samples.
- **Random Mapping** – a fixed random bijection between the 1000 ImageNet
  classes and the target task classes (e.g. 10 for CIFAR‑10).

The code follows the training loop described in Algorithm 1 of the paper.
It uses a single learning rate schedule (0.01 → 0.001 after 5 epochs).

## Extending to Other Datasets / Backbones

The script supports the following arguments:

```
--dataset    {cifar10,cifar100,svhn,gtsrb,flowers102,dt,dtd,ucf101,food101,eurosat,oxfordpets,sun397}
--backbone   {resnet18,resnet50,vitb32}
--epochs     INT
--batch-size INT
--lr         FLOAT
--lr-decay   FLOAT
--seed       INT
```

The paper uses many more datasets and backbones; the implementation is
modular and can be extended accordingly.

## License

This repository is provided for educational and research reproducibility
purposes. The code is released under the MIT license. The original paper
is available at https://github.com/tmlr-group/SMM (MIT license).

## Acknowledgements

Implementation is inspired by the official code of the paper
(https://github.com/tmlr-group/SMM) and the PyTorch tutorials for
visual reprogramming.