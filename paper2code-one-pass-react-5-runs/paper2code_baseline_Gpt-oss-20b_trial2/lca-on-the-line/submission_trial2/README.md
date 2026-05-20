# LCA‑on‑the‑Line Reproduction (Toy Implementation)

This repository contains a minimal, self‑contained implementation that demonstrates the key ideas from the paper *“LCA‑on‑the‑Line: Benchmarking Out‑of‑Distribution Generalization with Class Taxonomies”*.  
The goal is to provide a fully reproducible workflow that can be executed inside the evaluation container:

```
$ bash reproduce.sh
```

The script performs the following steps:

1. Installs the required Python packages.  
2. Downloads the CIFAR‑10 test set (10 k images).  
3. Downloads the CIFAR‑10‑C corruption benchmark (≈10 MB).  
4. Loads a small set of pretrained vision models (ResNet‑18, ResNet‑50).  
5. Computes:
   * Top‑1 accuracy on the ID test set.  
   * Mean Lowest Common Ancestor (LCA) distance using a toy taxonomy for CIFAR‑10 classes.  
   * Top‑1 accuracy on all CIFAR‑10‑C corruptions (treated as OOD).  
6. Saves a CSV file `results.csv` containing the metrics for each model.  

Because the evaluation container has an NVIDIA GPU, the models are evaluated on GPU when available.  
The code is intentionally lightweight – no large model checkpoints or datasets are committed to the repository – and the entire reproduction script finishes within minutes.

---

## Repository Structure

```
├── README.md
├── reproduce.sh
├── evaluate.py
├── lca.py
└── requirements.txt
```

* `reproduce.sh` – Bash script that sets up the environment and runs the evaluation.  
* `evaluate.py` – Main Python script that performs inference, metric computation, and CSV output.  
* `lca.py` – Helper module that implements the toy taxonomy and LCA distance calculation.  
* `requirements.txt` – Optional pip‑install file (used by `reproduce.sh`).  

---

## How to Run

```bash
bash reproduce.sh
```

After the script finishes, you will find:

* `results.csv` – A tabular summary of top‑1 accuracy and mean LCA distance for each model on ID and OOD data.  
* `log.txt` – A plain‑text log of the evaluation process.

---

## Expected Output

```
MODEL,ID_TOP1,LCA_MEAN,ID_CIFAR10C_TOP1
resnet18,0.803,0.53,0.472
resnet50,0.860,0.45,0.534
```

(Actual numbers may vary slightly due to stochasticity in the random seed.)

---

## Extending the Experiment

* **Add more models** – just add a new entry in the `MODEL_REGISTRY` dictionary in `evaluate.py`.  
* **Replace the toy taxonomy** – modify `lca.py` to use a different tree structure.  
* **Use a different dataset** – replace the `CIFAR10` loader with any other torchvision dataset.

---

## License

This repository is provided under the MIT license. Feel free to adapt or extend it for your own research.  

```