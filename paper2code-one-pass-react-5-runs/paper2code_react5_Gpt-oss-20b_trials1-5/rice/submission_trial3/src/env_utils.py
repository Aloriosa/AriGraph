"""
Utility functions for manipulating Gymnasium environments.
"""

def set_env_state(env, state):
    """
    Set the internal state of a Gymnasium environment that exposes a `state` attribute.
    Works for most classic control environments (e.g., CartPole, MountainCar).

    Parameters
    ----------
    env : gymnasium.Env
        The environment instance.
    state : np.ndarray
        State array to set.
    """
    # unwrap TimeLimit wrapper if present
    if hasattr(env, "env"):
        env = env.env
    if hasattr(env, "state"):
        env.state = state
    else:
        raise RuntimeError("The environment does not expose an internal `state` attribute.")