# Reproduction of “Fine‑tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem”

This repository contains a **minimal but faithful** implementation of the main ideas presented in the paper by Wołczyk et al. (ICML 2024).  
The code demonstrates:

* **Rule‑based data generation** – a deterministic policy that mimics a simple rule‑based agent (e.g. a left/right heuristic for `CartPole‑v1`).
* **Behavioural Cloning (BC)** – supervised learning on the offline dataset.
* **Fine‑tuning** – further training on the same environment using a simple actor–critic (REINFORCE with baseline).
* Optional **knowledge‑retention mechanisms**:
  * **Behavioural Cloning (BC)** – KL loss on a buffer of offline states.
  * **Kick‑starting (KS)** – KL loss on online data against the teacher.
  * **Elastic Weight Consolidation (EWC)** – regularisation that penalises changes to important weights.
  * **Episodic Memory (EM)** – replay of offline transitions during fine‑tuning.
* **Evaluation** of the policy after a fixed number of training steps and saving the results to `output.csv`.

> **Why CartPole?**  
> The original paper evaluates on complex domains such as NetHack, Montezuma’s Revenge, and robotic manipulation. Re‑implementing those environments would require far more code and compute. Instead, this repository uses the lightweight `CartPole‑v1` environment from OpenAI Gym as a stand‑in to illustrate the training pipeline and the effect of the different forgetting‑mitigation losses. The training curves and CSV output are sufficient to illustrate the effect of the different knowledge‑retention methods.

## Folder structure

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
└── train.py
```

## What this repository reproduces

* A **rule‑based dataset generation** stage that mimics a deterministic policy on `CartPole‑v1`.
* A **behavioural‑cloning pre‑training** stage that learns to mimic the rule‑based policy.
* A **fine‑tuning** stage where the pre‑trained policy is further trained on the same environment.
* Optional **knowledge‑retention** mechanisms (BC, KS, EWC, EM).
* An **evaluation loop** that runs the current policy for 10 episodes every `eval_interval` training steps and records the mean and standard‑deviation of the episode rewards in a CSV file.

## Running the reproduction

```bash
# ensure the script is executable
chmod +x reproduce.sh

# run the full pipeline
./reproduce.sh
```

The script will install dependencies, generate the offline dataset, pre‑train, fine‑tune and finally write `output.csv` containing the step, mean reward and standard‑deviation.

## Hyper‑parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--timesteps` | 200 000 | Total fine‑tuning steps |
| `--eval-interval` | 20 000 | Steps between evaluations |
| `--use_bc` | flag | Enable behavioural cloning loss |
| `--use_ks` | flag | Enable kick‑starting loss |
| `--use_ewc` | flag | Enable EWC regularisation |
| `--use_em` | flag | Enable episodic memory replay |
| `--seed` | 42 | Random seed for reproducibility |

Feel free to change the hyper‑parameters on the command line.