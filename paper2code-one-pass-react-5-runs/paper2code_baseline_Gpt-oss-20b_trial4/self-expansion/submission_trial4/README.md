# SEMA – Self‑Expansion of Pre‑trained Models with Mixture of Adapters

This repository contains a lightweight implementation of the **SEMA** continual learning method described in

> *Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning*  
> Huiyi Wang, Haodong Lu, Lina Yao, Dong Gong

The goal of the repository is to provide a **fully reproducible** end‑to‑end pipeline that can be executed in a fresh Ubuntu 24.04 Docker container with an NVIDIA A10 GPU.

> **Important**  
> Only source code and small helper files are committed.  
> The container will run `bash reproduce.sh` from the repository root and will produce a `results.txt` file containing the per‑task accuracy and the overall average accuracy.

---

## Repository Layout

```
/
├── dataset.py          # Simple dataset wrapper that creates class‑incremental tasks
├── models.py           # ViT backbone + adapters + representation descriptors
├── train_sema.py       # Main training script
├── utils.py            # Utility functions
├── reproduce.sh        # Shell script that installs deps, runs training, and prints summary
└── README.md
```

The implementation is intentionally lightweight:
* Only **CIFAR‑10** is used (10 classes).  
* The ViT‑B/16 backbone from `timm` is used and frozen.  
* Adapters are small 2‑layer MLPs with a hidden dimension of 8.  
* Representation descriptors are simple 2‑layer auto‑encoders with hidden dimension 8.  
* Training is limited to **1 epoch per task** (5 tasks) to keep runtime < 5 min on the A10.

Feel free to extend the code to use larger datasets or more epochs – the skeleton is ready for that.

---

## Running the Code

```bash
# 1.  Install the repository on a fresh container
git clone <this-repo-url>   # or copy the repository locally

# 2.  Make reproduce.sh executable
chmod +x reproduce.sh

# 3.  Run the reproduction script
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `timm`, `torchvision`).  
2. Download CIFAR‑10.  
3. Train SEMA on 5 incremental tasks.  
4. Write a `results.txt` file with per‑task accuracy and overall average accuracy.  
5. Print the same summary to the console.

---

## Expected Output

```
Task 1 accuracy: 82.4%
Task 2 accuracy: 71.9%
Task 3 accuracy: 70.2%
Task 4 accuracy: 68.7%
Task 5 accuracy: 66.3%
Average accuracy (last task): 68.3%
```

`results.txt` will contain the same lines. The numbers will vary slightly due to random initialization.

---

## Credits

This code is based on the official paper description.  
If you use it in a publication, please cite the paper and the original authors.

---

## Troubleshooting

| Symptom | Possible Fix |
|---------|--------------|
| `ModuleNotFoundError: No module named 'timm'` | Ensure `pip install timm` succeeded in `reproduce.sh`. |
| Training takes > 7 days | The script is set to 1 epoch per task; increase epochs only if you have enough GPU time. |
| GPU not recognized | The container must have NVIDIA drivers and the `nvidia-docker` runtime. |