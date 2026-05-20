import csv
import torch
from src.trainer import train

if __name__ == "__main__":
    # Use CPU for reproducibility; switch to 'cuda' if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = train(device=device)

    # Evaluate final policies
    final_rewards = agent.evaluate()
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "mean_reward"])
        for i, r in enumerate(final_rewards):
            writer.writerow([i, f"{r:.4f}"])
    print("Training complete. Results saved to results.csv")