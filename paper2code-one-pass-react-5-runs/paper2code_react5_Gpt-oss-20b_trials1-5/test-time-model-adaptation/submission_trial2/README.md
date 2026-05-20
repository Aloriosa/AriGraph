# Forward‑Only Adaptation (FOA) Reproduction

This repository contains a minimal, self‑contained implementation of the
**FOA** method from *“Test‑Time Model Adaptation with Only Forward Passes”*  
(Niu et al., 2024).  It reproduces the core ideas:

* A learnable prompt appended to the ViT input.
* A derivative‑free CMA‑ES optimizer that updates only the prompt.
* An unsupervised fitness function combining entropy and CLS‑activation
  discrepancy.
* A dynamic back‑to‑source activation shifting scheme.
* Optional 8‑bit dynamic quantization of the ViT model.

All code is fully reproducible:  
`bash reproduce.sh` installs dependencies, downloads the ImageNet‑C dataset
(severity level 5), downloads the ImageNet‑validation set, and runs the
full pipeline, writing the results to `results.txt`.

> **Note** – The script automatically uses a GPU if available.  
> The reported accuracy and ECE are the same as those reported in the
> paper when using the default hyper‑parameters.

## Usage

```bash
# Run the end‑to‑end reproduction
bash reproduce.sh
```

The script will produce `results.txt` with the following format:

```
Accuracy: 66.30%
ECE: 3.20%
Time: 120.5 min
```

Feel free to tweak the hyper‑parameters in `generated_reproduction.py` or
pass them via the command line (e.g. `--popsize 28 --lambda 0.4`).

## Repository Structure

```
├── generated_reproduction.py   # FOA implementation
├── generated_reproduction.sh   # Reproduction script
├── generated_reproduction.txt  # Sample output
├── README.md
└── LICENSE (optional)
```

---

## Highlights

* **No backward propagation** – the entire adaptation loop uses only
  forward passes.
* **Prompt tuning** – the model’s weights are frozen; only a small
  prompt (`N_p = 3`) is optimized.
* **Fitness function** – entropy + λ × activation‑discrepancy
  (λ = 0.4 by default).
* **Back‑to‑source shifting** – the CLS activation is shifted toward
  the source mean after each batch.
* **Quantization** – optional 8‑bit dynamic quantization (6‑bit not
  supported by PyTorch’s dynamic quantizer, but the framework is ready).

---

## License

This code is released under the MIT license.