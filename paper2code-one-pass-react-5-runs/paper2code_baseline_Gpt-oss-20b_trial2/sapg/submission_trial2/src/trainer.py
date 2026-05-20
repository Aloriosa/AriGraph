import time
from tqdm import tqdm
from .agent import SAPGAgent


def train(env_name='Pendulum-v1',
          epochs=5,
          M=3,
          envs_per_policy=4,
          horizon=16,
          seed=42,
          device='cpu'):
    torch.manual_seed(seed)
    agent = SAPGAgent(env_name, M=M, envs_per_policy=envs_per_policy,
                      horizon=horizon, device=device)

    for epoch in range(1, epochs + 1):
        start = time.time()
        trajs = agent.collect_data()
        agent.update(trajs)
        rewards = agent.evaluate()
        print(f"Epoch {epoch}/{epochs} | "
              f"Policy 0 reward: {rewards[0]:.2f} | "
              f"Policy 1 reward: {rewards[1]:.2f} | "
              f"Policy 2 reward: {rewards[2]:.2f} | "
              f"t={time.time() - start:.1f}s")
    return agent