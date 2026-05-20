The `componet/` package implements the core *Self‑Composing Policies* architecture.

- `module.py` – single self‑composing module (attention heads + MLP).
- `policy.py` – custom SB3 policies that stack modules and freeze previous ones.