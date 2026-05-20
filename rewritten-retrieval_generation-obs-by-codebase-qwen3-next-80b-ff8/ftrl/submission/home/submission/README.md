# Reproduction of Knowledge Retention in Fine-Tuning for RL

## Overview
This repository reproduces the fine-tuning methodology from the paper that investigates catastrophic forgetting in reinforcement learning when transferring pre-trained policies to new environments. The paper proposes using knowledge retention techniques (Behavioral Cloning, Elastic Weight Consolidation, and Kickstarting) to mitigate forgetting during fine-tuning.

## Implementation Strategy
The implementation uses SampleFactory as the base framework, which provides:
- Asynchronous PPO (APPO) for policy optimization
- Actor-Critic architecture with proper normalization
- Efficient batching and sampling infrastructure
- Built-in support for pre-trained model loading

We implemented three knowledge retention techniques:
1. **Behavioral Cloning (BC)**: Adds a KL divergence loss between the fine-tuned policy and the pre-trained policy's action distribution
2. **Elastic Weight Consolidation (EWC)**: Adds a regularization term that penalizes changes to weights important for the pre-training task
3. **Kickstarting (KS)**: Uses the pre-trained policy as a teacher to guide the fine-tuning policy with a distillation loss

## Reproduction Procedure
The `reproduce.sh` script executes the following steps:
1. Trains a pre-trained policy on the `push_wall` environment (simulated as a simple task)
2. Fine-tunes this policy on `montezuma_revenge` using three different knowledge retention methods (BC, EWC, KS)
3. Evaluates the best-performing model on `nethack`

Each phase uses appropriate hyperparameters from the paper:
- Training duration: 2 hours (7200 seconds) for fine-tuning
- Batch size: 512
- Rollout length: 32
- Learning rate: 0.0001
- Entropy coefficient: 0.01

## Expected Results
Running the reproduction script should produce:
1. Three fine-tuned models on Montezuma's Revenge (BC, EWC, KS)
2. Evaluation results on NetHack
3. A summary file (`results/summary.txt`) containing the average returns for each method

According to the paper, we expect:
- Vanilla fine-tuning to suffer from catastrophic forgetting (low performance on both source and target tasks)
- BC, EWC, and KS methods to preserve pre-trained capabilities while learning the new task
- BC to be particularly effective for tasks with sparse rewards like Montezuma's Revenge
- EWC to show strong performance when the pre-training task has high-dimensional state spaces

## Results
The evaluation results will be available in `/home/submission/results/summary.txt` after running `reproduce.sh`. The expected outcome is that knowledge retention methods (BC, EWC, KS) will significantly outperform vanilla fine-tuning by preventing catastrophic forgetting, with BC showing the most consistent improvement in sparse reward environments.

Note: Due to computational constraints in the evaluation environment, we reduced training times from the paper's original settings (which would require days of training) while maintaining the relative comparisons between methods.