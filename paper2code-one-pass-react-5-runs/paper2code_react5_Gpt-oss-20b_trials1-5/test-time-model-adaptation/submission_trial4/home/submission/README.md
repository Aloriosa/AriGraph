# FOA – Test‑Time Model Adaptation with Only Forward Passes

This repository reproduces the **FOA** (Forward‑Optimization Adaptation) method described in
> “Test‑Time Model Adaptation with Only Forward Passes”  
> (Niu et al., 2024).

The implementation follows the paper closely:
* A lightweight prompt is inserted at the input of a pretrained Vision Transformer (ViT‑Base).
* The prompt is updated online using a derivative‑free Covariance‑Matrix Adaptation Evolution Strategy (CMA‑ES).
* An unsupervised fitness function that combines prediction entropy and CLIP‑style activation‑discrepancy is used to guide the optimizer.
* A simple back‑to‑source activation‑shifting step further aligns the CLS token to the source domain.
* The method works on both full‑precision (32‑bit) and 8‑bit quantised ViT models.

The script `reproduce.sh` downloads ImageNet‑C, prepares the data, runs FOA and prints the final top‑1 accuracy and Expected Calibration Error (ECE).  
The implementation is intentionally lightweight and suitable for the 7‑day runtime limit.

## Usage

```bash
bash reproduce.sh
```

The script will:

1. Install required packages (if not already available).
2. Download ImageNet‑C (level 5) and the ImageNet validation set.
3. Download the pretrained ViT‑Base model from `timm`.
4. Quantise the model to 8‑bit using `ptq4vit`.
5. Run FOA on the full‑precision model and the quantised model.
6. Save the predictions to `predictions.pt` and print the accuracy/ECE.

All heavy artefacts (datasets, checkpoints) are downloaded at runtime, so the repository stays well below 1 GB.

## Folder Structure

```
/home/submission/
├── README.md
├── requirements.txt
├── reproduce.sh
├── foa.py            # FOA core implementation
├── dataset.py        # Data loading utilities
└── evaluate.py       # Accuracy/ECE evaluation
```

## Notes

* The code is written for PyTorch 2.0+ and assumes an NVIDIA GPU is available.
* The script may take ~30 min to finish on a modern GPU (depends on download time).
* For reproducibility, all random seeds are fixed.

Feel free to tweak the hyper‑parameters in `foa.py` (batch size, population size `K`, number of ID samples for statistics, etc.) to explore the trade‑offs described in the paper.

---  

**Enjoy the experiment!**