"""
Main script that runs the three baselines and prints the results.
"""

import torch
from apt.trainer import train_baseline
from apt.config import TrainingConfig

def main():
    config = TrainingConfig()
    results = []

    print("=== Training LoRA baseline ===")
    results.append(train_baseline(config, baseline="lora"))

    print("\n=== Training Prune baseline ===")
    results.append(train_baseline(config, baseline="prune"))

    print("\n=== Training APT (adaptive) ===")
    results.append(train_baseline(config, baseline="apt"))

    # Pretty print results
    print("\n=== Final Results ===")
    for r in results:
        print(f"\nBaseline: {r['baseline']}")
        print(f"  Accuracy              : {r['accuracy']:.4f}")
        print(f"  Training time (s)     : {r['train_time_sec']:.1f}")
        print(f"  Peak GPU memory (MB)  : {r['peak_gpu_mem_MB']:.1f}")
        print(f"  Inference time (s)    : {r['inference_time_sec']:.1f}")

if __name__ == "__main__":
    main()