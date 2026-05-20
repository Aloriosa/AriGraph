import numpy as np

class RewardFunction():
    # Given a batch of trajectory states and random states, generate a reward function.
    # Return the labelled state-reward pairs. (batch_size, num_pairs, obs_dim + 1)
    def generate_params_and_pairs(self, traj_states, random_states):
        raise NotImplementedError
    
    # Given a batch of states and a batch of parameters, compute the reward.
    def compute_reward(self, states, params):
        raise NotImplementedError