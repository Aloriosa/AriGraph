# Simformer – All‑in‑One Simulation‑Based Inference (Reproduction)

This repository contains a lightweight, fully reproducible implementation of the core ideas from the *All‑in‑One Simulation‑Based Inference* paper (Simformer).  
The implementation focuses on the following key components:

1. **Tokenizer** – each variable (parameter or data) is represented by an identifier embedding, a value embedding, and a binary conditional state.  
2. **Transformer** – a standard multi‑head self‑attention network that accepts a sequence of tokens and an optional structured attention mask.  
3. **Diffusion model** – a Variance‑Exploding SDE (VESDE) with denoising score matching.  
4. **Joint training** – the model is trained on the joint distribution \(p(\theta, x)\) with a random condition mask, allowing the network to learn *all* conditional distributions.  
5. **Arbitrary conditioning** – arbitrary posterior, likelihood, and parameter conditionals are sampled by fixing the corresponding tokens during reverse diffusion.  
6. **Guided diffusion** – a simple interval‑based classifier‑free guidance is provided as an example of how custom constraints can be imposed during sampling.  

The repository contains two benchmark tasks from the paper:

* **Gaussian‑Linear** – a 20‑dimensional joint distribution (10‑dimensional parameters + 10‑dimensional data).  
* **Two‑Moons** – a 4‑dimensional joint distribution (2‑dimensional parameters + 2‑dimensional data).

Running the provided `reproduce.sh` script will:

1. Install the required Python packages.  
2. Train the Simformer on both tasks.  
3. Generate posterior samples for a held‑out observation.  
4. Save the results in the `results/` directory as CSV files.

> **Note** – The full Simformer implementation in the original paper includes many additional features (function‑valued parameters, missing data handling, complex attention masks, extensive guidance, etc.).  For the purpose of this reproduction task, the implementation focuses on the core workflow while still demonstrating all key concepts.

---

## Repository Structure

```
/home/submission
├── README.md
├── reproduce.sh
├── requirements.txt
├── src
│   ├── tokenizer.py
│   ├── transformer_model.py
│   ├── diffusion.py
│   ├── model.py
│   ├── train.py
│   ├── sample.py
│   └── utils.py
├── tasks
│   ├── gaussian_linear.py
│   └── two_moons.py
└── results
    └── (generated)
```

---

## Reproduction

```bash
bash reproduce.sh
```

The script will run for < 5 min on a typical CPU.  
The final output files are:

* `results/gaussian_linear_posterior.csv` – posterior samples for the Gaussian‑Linear task.  
* `results/gaussian_linear_predictive.csv` – posterior‑predictive samples.  
* `results/two_moons_posterior.csv` – posterior samples for the Two‑Moons task.  
* `results/two_moons_predictive.csv` – posterior‑predictive samples.

Each CSV contains the sampled parameters followed by the sampled data (when applicable).

---

## Customisation

* **Attention masks** – See `src/utils.py` for helper functions to construct masks that encode simulator dependencies.  
* **Guidance** – The `sample.py` module implements simple interval‑based guidance. Feel free to extend it with more sophisticated strategies.  
* **Function‑valued parameters** – The tokenizer can embed a vector of values as a *single* token. See `src/tokenizer.py` for the implementation.  
* **Missing data** – Missing entries can be encoded by setting the corresponding conditional flag to `1` (conditioned) and passing a dummy value (e.g. 0).  

---

## Limitations

* The implementation focuses on the core ideas and does **not** reproduce the full experimental suite of the paper (e.g. Lotka‑Volterra, SIRD, Hodgkin‑Huxley, gravitational waves).  
* Baseline comparisons (NPE, NLE, etc.) and detailed quantitative metrics are omitted for brevity.  
* The diffusion guidance is a simple illustration and does not cover the full range of constraints described in the paper.

---

## Contact

For questions or suggestions, please open an issue or contact the original authors.