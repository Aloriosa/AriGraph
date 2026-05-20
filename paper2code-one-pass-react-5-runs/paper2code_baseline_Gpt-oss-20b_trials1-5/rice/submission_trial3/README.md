# RICE – Refining Reinforcement Learning with Explanation (Reproduction)

This repository contains a lightweight, fully‑reproducible implementation of the core ideas presented in  
*RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation*  
(Cheng et al., 2024).  

The code follows the same high‑level pipeline as the paper:

1. **Pre‑training** – Train a baseline PPO agent on a simple environment (`CartPole-v1`).  
2. **Explanation** – Identify *critical states* from the pre‑trained policy using a simple advantage‑based
   heuristic (substitutes the full StateMask mask network for brevity).  
3. **Refinement** – Re‑train a new PPO agent starting from a **mixed initial state distribution**  
   (default initial states + sampled critical states) while adding a **Random Network Distillation (RND)**
   intrinsic bonus to encourage exploration.  
4. **Evaluation** – Run the refined policy for a few episodes and report the average return.

The entire reproduction can be executed with a single shell script `reproduce.sh`.  
No heavy data or checkpoints are committed – the models are re‑trained from scratch during the
run.  
The training times are modest (≈ 5 min on a CPU) and the entire repository is well below the
1 GB limit.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.  
2. Train the baseline PPO agent (`src/train_pretrained.py`).  
3. Generate critical states (`src/train_critical.py`).  
4. Perform refinement with mixed initial states and RND (`src/refine.py`).  
5. Print the average return of the refined policy after 10 evaluation episodes.

All intermediate artefacts (trained models, critical state list, logs) are stored in the
`results/` directory.

## Repository Structure

```
src/
├── train_pretrained.py   # Baseline PPO training
├── train_critical.py     # Critical state extraction
├── refine.py             # Refinement training (mixed init + RND)
├── rnd.py                # Random Network Distillation implementation
├── utils.py              # Helper functions
├── mixed_reset_env.py    # Wrapper for mixed initial state distribution
├── rnd_wrapper.py        # Wrapper that adds intrinsic reward & trains RND
└── constants.py          # Hyper‑parameters
reproduce.sh
README.md
requirements.txt
```

## Expected Output

After running `reproduce.sh` you should see output similar to:

```
Training baseline PPO...
... 200000 steps completed ...
Saving baseline model to results/baseline.zip

Extracting critical states...
Saved 200 critical states to results/critical_states.npy

Refining policy with mixed initial states and RND...
... 200000 steps completed ...
Saving refined model to results/refined.zip

Evaluating refined policy over 10 episodes...
Average return: 195.3
Reproduction finished.
```

The refined policy should achieve a higher average return than the baseline,
demonstrating the benefit of the RICE refinement scheme.

## Notes

* The implementation uses the `CartPole-v1` environment for simplicity – the same
  pipeline can be applied to any gymnasium environment that allows manual state
  setting (e.g. custom wrappers).  
* The critical‑state extraction is a lightweight surrogate for the full
  StateMask explanation and is sufficient to illustrate the RICE concepts.  
* The RND intrinsic reward uses a fixed random target network and a trainable
  predictor; both are updated online during refinement.

Feel free to experiment with different hyper‑parameters (`p`, `lambda`, `alpha`,
etc.) by editing `constants.py`.