The `experiments/` folder contains the two training scripts used in the reproduction:

- `meta_world.py` – trains SAC on 20 Meta‑World tasks.
- `ale.py` – trains PPO on 17 Atari tasks (SpaceInvaders + Freeway).

Both scripts use the custom `CompoNetActorCriticSAC` / `CompoNetActorCriticPPO` policies defined in the `componet` package.