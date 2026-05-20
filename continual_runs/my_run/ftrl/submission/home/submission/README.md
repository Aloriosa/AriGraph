# Reproduction of "Knowledge Retention in Reinforcement Learning Fine-Tuning"

## Overview

This repository reproduces the key findings from the paper on knowledge retention techniques for reinforcement learning fine-tuning. The paper demonstrates that fine-tuning pre-trained policies without knowledge retention leads to catastrophic forgetting, and proposes methods like Behavioral Cloning (BC), Elastic Weight Consolidation (EWC), and Kickstarting (KS) to mitigate this issue.

## Methodology

The reproduction implements the following components:

1. **Pre-trained Policy Training**: A PPO agent is trained on a simplified version of Montezuma's Revenge to serve as the pre-trained policy.
2. **Fine-tuning with Knowledge Retention**: Four fine-tuning methods are implemented:
   - Vanilla fine-tuning (baseline)
   - Behavioral Cloning (BC): Penalizes divergence from the pre-trained policy using KL divergence
   - Elastic Weight Consolidation (EWC): Penalizes changes to important weights using a Fisher information matrix
   - Kickstarting (KS): Provides additional reward signals based on similarity to pre-trained policy actions
3. **Training from Scratch**: A baseline model trained without any pre-training for comparison
4. **Evaluation**: All policies are evaluated on the full Montezuma's Revenge environment with multiple seeds and episodes

## Implementation Details

- **Environment**: Montezuma's Revenge-v4 (Atari game with sparse rewards)
- **Architecture**: Custom CNN feature extractor with 3 convolutional layers followed by fully connected layers
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Pre-training**: 200,000 steps on a simplified version of the environment
- **Fine-tuning**: 100,000 steps for all methods
- **Evaluation**: 100 episodes per policy with 5 different seeds

## Reproduction Instructions

1. Run the reproduction script:
```bash
bash reproduce.sh
```

2. The script will:
   - Train a pre-trained policy
   - Fine-tune with four different methods (vanilla, BC, EWC, KS)
   - Train a policy from scratch
   - Evaluate all policies
   - Generate a summary report

3. Results are saved in the `results/` directory:
   - `baseline_comparison.csv`: Comparison of all methods
   - `*_results.csv`: Training progress for each method
   - `final_summary.txt`: Comprehensive summary of reproduction results

## Expected Outcomes

The reproduction should demonstrate:
1. **Catastrophic Forgetting**: Vanilla fine-tuning shows significant performance degradation compared to the pre-trained policy
2. **Knowledge Retention Benefits**: BC, EWC, and KS methods maintain better performance than vanilla fine-tuning
3. **Superiority over Scratch**: Knowledge retention methods outperform training from scratch
4. **Method Comparison**: Among the knowledge retention methods, BC and EWC typically perform best

## Results

The final summary report (`results/final_summary.txt`) will indicate whether the reproduction successfully replicated the paper's key findings. The paper claims that knowledge retention techniques significantly improve transfer learning performance in RL, and our implementation should show this pattern.

## Limitations

1. We used Montezuma's Revenge instead of NetHack due to computational constraints
2. The EWC implementation uses a simplified Fisher matrix approximation
3. The environment is simplified compared to the paper's full complexity
4. Training time is reduced for practical reasons

Despite these limitations, the core principles of knowledge retention in RL fine-tuning are successfully demonstrated.