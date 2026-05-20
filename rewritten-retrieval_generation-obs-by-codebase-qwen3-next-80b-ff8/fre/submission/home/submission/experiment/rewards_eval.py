import numpy as np

class VelocityRewardFunction():
    def __init__(self):
        pass
    
    # Select an XY velocity from a future state in the trajectory.
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        selected_traj_state_idx = np.random.randint(traj_states.shape[1], size=(batch_size,))
        selected_traj_state = traj_states[np.arange(batch_size), selected_traj_state_idx] # (batch_size, obs_dim)
        params = selected_traj_state[:, 15:17] # (batch_size, 2)
        params[:batch_size//4] = np.random.uniform(-1, 1, size=(batch_size//4, 2)) # Randomize 25% of the time.
        params = params / np.linalg.norm(params, axis=-1, keepdims=True) # Normalize XY

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def compute_reward(self, states, params):
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        xy_vels = states[..., 15:17] * 0.33820298
        return np.sum(xy_vels * params, axis=-1) # (batch_size,)
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)