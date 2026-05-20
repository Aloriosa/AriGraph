# RICE Policy Refinement Reproduction

This repository reproduces the RICE (Refined Initial State Exploration) algorithm from the paper "RICE: Refined Initial State Exploration for Deep Reinforcement Learning" (Cheng et al., 2023).

## Overview

RICE is a policy refinement technique that improves sample efficiency and prevents premature convergence in DRL by using explanation methods to identify "critical states" from a pre-trained policy's trajectory history. These critical states are then used to construct a mixed initial state distribution that combines default initialization with critical states, enabling more effective exploration.

## Implementation Details

### Core Components

1. **Pre-trained Policy**: We train a PPO agent on Mujoco environments as the guide policy
2. **StateMask Explanation Method**: We implement Integrated Gradients to identify critical states with high explanatory significance
3. **Mixed Initial State Distribution**: We create a mixture of default states and critical states using a configurable mixing ratio
4. **Policy Refinement**: We use the mixed distribution to reset the environment during training, refining the policy without retraining from scratch

### Key Paper Requirements Implemented

- ✅ Uses pre-trained DRL policy (PPO) as guide
- ✅ Implements StateMask explanation method using Integrated Gradients
- ✅ Identifies critical states as top-k states with highest saliency scores
- ✅ Constructs mixed initial state distribution (default + critical states)
- ✅ Refines policy by resetting environment to mixed distribution
- ✅ Evaluates on Mujoco environments (HalfCheetah-v4)
- ✅ Compares against baseline methods
- ✅ Uses 5 seeds for evaluation as specified in paper
- ✅ Reports mean and standard deviation of performance

### Configuration Parameters

- Mixing ratio of default to critical states: 0.5 (50%)
- Top-k critical states: 10% of trajectory steps
- Training steps for pre-training: 300,000
- Training steps for refinement: 100,000
- Evaluation episodes: 100 per seed
- Number of evaluation seeds: 5

## Reproduction Results

Running `reproduce.sh` will:
1. Train a PPO policy on HalfCheetah-v4 (300k steps)
2. Extract critical states using Integrated Gradients
3. Refine the policy using RICE (100k steps)
4. Compare against baselines (no refinement, random sampling, pure critical states)

Expected outcomes:
- RICE should achieve higher mean episode return than baseline PPO
- RICE should show improved sample efficiency (faster convergence)
- Performance improvement should be statistically significant (p < 0.05 via t-test)
- The critical states visualization should show states near the "frontier" of the policy's exploration

## File Structure

- `reproduce.sh`: Main reproduction script
- `generate_trajectories.py`: Generates offline trajectories from pre-trained policy
- `train_ppo.py`: Trains the base PPO policy
- `extract_critical_states.py`: Implements StateMask explanation method
- `refine_policy_rice.py`: Implements the RICE algorithm
- `evaluate_baselines.py`: Compares RICE against baselines
- `visualize_critical_states.py`: Creates visualization of critical states
- `models/`: Stores trained policy weights
- `data/`: Stores trajectory and critical state data
- `results/`: Stores evaluation results and visualizations

## Dependencies

- Python 3.8+
- PyTorch with CUDA support
- Gymnasium with Mujoco environments
- NumPy, Matplotlib, Scikit-learn

## Limitations

This implementation assumes the environment has reset capability to any previously visited state, as required by RICE. The paper mentions evaluation on multiple environments, but we focus on HalfCheetah-v4 as a representative Mujoco task. The implementation uses Integrated Gradients as the explanation method as specified in the paper, though other methods like attention weights could be substituted.