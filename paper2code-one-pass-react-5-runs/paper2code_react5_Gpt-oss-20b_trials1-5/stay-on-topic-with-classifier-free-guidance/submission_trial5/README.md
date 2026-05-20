# Stay on topic with Classifier‑Free Guidance

This repository contains a minimal, reproducible implementation of the
Classifier‑Free Guidance (CFG) technique described in the paper
*“Stay on topic with Classifier‑Free Guidance”*.  The goal is to provide
an end‑to‑end pipeline that demonstrates the key ideas of the paper
without requiring large computational resources.

## Features

| Feature | Description |
|---------|-------------|
| **CFG inference** | Implements Eq. 7 of the paper for any causal LM (GPT‑2, GPT‑Neo, etc.). |
| **Negative prompting** | Allows a user‑defined prompt that is *de‑prioritised* during generation. |
| **Benchmark** | Small LAMBADA benchmark: vanilla vs. CFG (γ = 1.5). |
| **Chain‑of‑Thought demo** | Simple arithmetic CoT example using CFG. |
| **Assistant demo** | System + user prompt with negative prompting. |

All code is self‑contained and requires only the public GPT‑2 medium
model, which is downloaded automatically by the scripts.

## Reproduction

The provided `reproduce.sh` script performs the following steps:

1. Installs Python 3 and `pip`.
2. Installs `torch`, `transformers`, `datasets`, and `tqdm`.
3. Runs three demo scripts:
   * `benchmark_lambada.py` – prints vanilla vs. CFG accuracy on a 200‑example subset.
   * `chain_of_thought_demo.py` – shows a CFG‑enhanced CoT generation.
   * `assistant_demo.py` – shows a system‑prompt + user‑prompt example with negative prompting.

Run the script from the repository root:

```bash
bash reproduce.sh
```

The script will output results directly to the console; no large artifacts are
generated or stored.

## Repository structure

```
.
├── cfg_inference.py          # Core CFG implementation
├── benchmark_lambada.py      # Small accuracy benchmark
├── chain_of_thought_demo.py  # CoT example
├── assistant_demo.py         # Assistant + negative prompt example
├── reproduce.sh              # End‑to‑end reproduction script
├── README.md
└── .gitignore
```

## Limitations

* The benchmark uses only 200 validation examples of LAMBADA for speed.
  The paper reports results on thousands of examples; reproducing the exact
  numbers would require a full dataset and more compute.
* Only GPT‑2‑medium is used.  Extending to other models (Pythia, LLaMA, etc.)
  is straightforward but not shown here.
* Temperature, top‑k/p, and γ are set to the values used in the paper’s
  qualitative examples (γ = 1.5 or 2.0).  A systematic sweep is omitted for brevity.

Despite these simplifications, the repository demonstrates all core
methodological contributions of the paper and can be extended to
full‑scale experiments if needed.

## License

MIT License – feel free to adapt for your own research or teaching
purposes.