# Test‑Time Forward‑Optimization Adaptation (FOA)

This repository reproduces the core idea of the **Test‑Time Model Adaptation with Only Forward Passes** paper.  
The implementation focuses on the following:

1. **Forward‑only adaptation** – a prompt is added to the input of a ViT‑Base model and updated using a covariance‑matrix adaptation evolution strategy (CMA‑ES).
2. **Unsupervised fitness** – the CMA optimisation is guided by a combination of prediction entropy and a simple activation‑discrepancy term.
3. **Activation shifting** – the CLS token of the transformer is moved towards the mean of a small source sample set.

The code is intentionally lightweight so that it can run in a short time on a CPU or GPU.  
It uses only the CIFAR‑10 dataset (≈125 MB) and the pre‑trained ViT‑Base weights from `timm`. No large model checkpoints or ImageNet data are included in the repository.

## What the script does

- Downloads the CIFAR‑10 dataset.
- Computes *source* CLS statistics from 32 training images.
- Generates a test batch of 64 images from the CIFAR‑10 test set.
- Runs FOA for 5 CMA iterations to adapt the prompt.
- Applies a single activation‑shifting step.
- Prints the top‑5 predictions for each test image and saves them to `output.json`.

## Expected output

After running `bash reproduce.sh` you should see a short progress bar, followed by a message:

```
Adaptation finished. Predictions written to output.json
```

The `output.json` file contains a list of 64 dictionaries, each with the image index and the top‑5 predicted class indices.

## How to extend

- Increase the number of CMA iterations (`cma_iters`) or population size (`k_pop`) for a more thorough adaptation.
- Replace the synthetic source statistics with real ImageNet samples for a faithful reproduction of the paper’s results.
- Swap the pre‑trained model (`vit_base_patch16_224`) for a quantised version if you have the weights.

## License

MIT License. Feel free to adapt or extend.