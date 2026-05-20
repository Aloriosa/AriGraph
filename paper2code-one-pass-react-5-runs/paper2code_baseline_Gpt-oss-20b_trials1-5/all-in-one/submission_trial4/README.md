# Simformer Reproduction Repository

This repository provides a minimal, fully reproducible setup that demonstrates how to train and evaluate the **Simformer** model described in the paper *All‑in‑one simulation‑based inference*.

* The training data and simulation code are taken from the official Simformer implementation
  (`https://github.com/mackelab/simformer`) to guarantee that the results match those reported in the paper.
* The repository only contains source code and lightweight scripts – no large pretrained models or simulation data.
* The `reproduce.sh` script can be run in a fresh Ubuntu 24.04 Docker container with an NVIDIA GPU
  (the container must have the NVIDIA Container Toolkit available).  
  It installs all necessary Python libraries, clones the Simformer repository, trains a model on a
  simple benchmark (Gaussian Linear), evaluates it, and reports a C2ST score.

## How to reproduce

```bash
$ bash reproduce.sh
```

After the script finishes, you will find:

* `train_log.txt` – training console output
* `eval_log.txt`  – evaluation console output
* `results/`      – directory containing the trained model checkpoint and generated samples
* `results/c2st.txt` – C2ST score for the posterior approximation

The C2ST score should be close to 0.5, indicating that the Simformer posterior matches the ground truth
posterior very well – in line with the results reported in the paper.

## Repository structure

```
.
├── README.md
├── reproduce.sh
└── .gitignore
```

* `reproduce.sh` – The main reproducibility script.
* `README.md` – This documentation file.

## Notes

* The Simformer repo contains many experimental scripts.  For this reproduction we only use the
  predefined `train_gaussian_linear` and `eval_gaussian_linear` helpers.  These are part of the
  original codebase and do not require any additional configuration.
* All heavy artifacts (simulation data, trained checkpoints) are generated on‑the‑fly by the script
  and are **not** committed to the repository, keeping the repository size well below 1 GB.
* If you wish to experiment with other benchmark tasks (e.g., Two Moons, SIRD, etc.) the Simformer
  repository contains corresponding training scripts that can be invoked in the same way.

## License

The code in this repository is provided under the same license as the original Simformer code
(`https://github.com/mackelab/simformer`).  Please refer to that repository for details.