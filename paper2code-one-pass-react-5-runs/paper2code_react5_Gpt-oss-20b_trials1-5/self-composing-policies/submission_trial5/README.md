# CompoNet – Self‑Composing Policies for Continual Reinforcement Learning

This repository reproduces the main ideas of the paper  
*“Self‑Composing Policies for Scalable Continual Reinforcement Learning”* by Mikel Malagón et al.  
The code implements a lightweight version of the **CompoNet** architecture and a simple
continual learning pipeline that trains on a sequence of OpenAI Gym tasks
(`CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`).  
The training uses a vanilla REINFORCE policy‑gradient algorithm;  
the focus is on demonstrating the *module addition, freezing and compositional*
behaviour rather than on achieving state‑of‑the‑art scores.

## Repository layout
```
.
├── requirements.txt
├── README.md
├── reproduce.sh
└── src
    ├── composenet_policy.py   # CompoNet implementation
    └── train_componet.py      # Continual training script
```

## How to run
```bash
bash reproduce.sh
```

The script installs the dependencies, trains the agent on the task sequence
and saves a `results.json` file with the final average return per task.
The final printout shows the per‑task performance and a simple overall
summary.

## What the code does
1. **CompoNet architecture** – each *module* contains:
   * an *output attention head* that linearly combines the previous modules’
     logits;
   * an *input attention head* that gathers context from the previous logits
     and the output head;
   * an *internal policy* (small MLP) that refines the final logits.
2. **Continual learning loop** – for each task:
   * add a new module, freeze all older ones;
   * train the new module with REINFORCE for a fixed number of episodes;
   * evaluate the agent on 5 episodes and record the return.
3. **Evaluation** – the script prints the per‑task average return and
   writes `results.json`.

The implementation follows the paper’s description while staying lightweight
and fully reproducible in a short training session.