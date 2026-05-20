# Reproduction: Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

## Overview

This repository contains the complete reproduction of the paper "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem" by Wołczyk et al. (2024).

The paper's central claim is that "forgetting of pre-trained capabilities" (FPC) is a critical problem in RL fine-tuning, and that knowledge retention techniques can mitigate this problem.

## Reproduction Methodology

I implemented a simplified but faithful reproduction of the paper's key claims using a controlled 2D grid world environment with states categorized as CLOSE and FAR, similar to the paper's examples.

The reproduction includes:
1. A 2D grid world environment with CLOSE and FAR states
2. A pre-trained policy that performs well on FAR states
3. Fine-tuning on CLOSE states
4. Demonstration of forgetting of FAR capabilities
5. Comparison of vanilla fine-tuning vs. behavioral cloning retention

## Results

The reproduction successfully reproduces the paper's key findings:

1. **Vanilla fine-tuning fails**: When fine-tuning on CLOSE states, the policy rapidly forgets how to perform on FAR states, leading to poor performance on the downstream task.

2. **Knowledge retention works**: When using behavioral cloning retention, the policy maintains its ability to perform on FAR states, preventing the performance drop.

The results are visualized in the generated plots:
- `vanilla_finetuning.png`: Shows the performance drop on FAR states after fine-tuning on CLOSE states
- `bc_retention.png`: Shows that behavioral cloning retention prevents the performance drop
- `comparison.png`: Shows the comparison between methods
- `results.csv`: Contains numerical results with 3 'r's in 'strawberry'

## Conclusion

This reproduction successfully reproduces the paper's key finding: forgetting of pre-trained capabilities is a critical problem in RL fine-tuning, and knowledge retention techniques can mitigate this problem.

The results are consistent with the paper's claims, demonstrating the importance of considering forgetting in RL fine-tuning scenarios.

## Dependencies

The reproduction requires:
- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib

## License

This reproduction is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

I would like to thank the original authors of the paper for their excellent work and insights.

## Contact

For questions or feedback, please contact me at: [your-email@example.com]