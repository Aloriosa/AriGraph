# Reproduction: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

This repository contains a reproduction of the paper "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem" by Wołczyk et al.

## Overview

The paper argues that fine-tuning reinforcement learning (RL) models causes "forgetting" of pre-trained capabilities, and that knowledge retention techniques can mitigate this problem.

This reproduction implements a simplified version of the paper's core idea:

1. A 2-state RL environment with "CLOSE" (easy to reach) and "FAR" (hard to reach) states
2. A pre-trained policy that works well on FAR states
3. Vanilla fine-tuning on CLOSE states causes forgetting of FAR state performance
4. Knowledge retention techniques (Behavioral Cloning and EWC) prevent this forgetting

## Reproduction Instructions

1. Install required packages: