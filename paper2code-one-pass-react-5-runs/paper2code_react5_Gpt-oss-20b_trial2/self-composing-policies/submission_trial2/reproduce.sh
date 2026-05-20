#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------
if ! command -v pip &>/dev/null; then
    echo "pip not found, aborting."
    exit 1
fi

echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------
# 2. Run training pipelines
# ------------------------------------------------------
echo "=== Training Meta‑World (SAC) ==="
python3 train_meta_world.py > meta_world.log 2>&1 || { echo "Meta‑World training failed"; exit 1; }

echo "=== Training ALE (SpaceInvaders) (PPO) ==="
python3 train_ale.py --env SpaceInvaders-v5 > spaceinvaders.log 2>&1 || { echo "SpaceInvaders training failed"; exit 1; }

echo "=== Training ALE (Freeway) (PPO) ==="
python3 train_ale.py --env Freeway-v5 > freeway.log 2>&1 || { echo "Freeway training failed"; exit 1; }

# ------------------------------------------------------
# 3. Aggregate results
# ------------------------------------------------------
echo "=== Aggregating results ==="
python3 - <<'PY'
import json, os, glob, re

def parse_log(file):
    # Very lightweight parser – extracts the last success rate line
    with open(file) as f:
        lines = f.readlines()
    for line in reversed(lines):
        m = re.search(r"Task (\d+): Success rate = ([0-9.]+)", line)
        if m:
            return int(m.group(1)), float(m.group(2))
    return None, None

results = {}

# Meta‑World
meta_res = []
for log in sorted(glob.glob("meta_world.log")):
    task, rate = parse_log(log)
    if task is not None:
        meta_res.append(rate)
results["MetaWorld"] = {"success_rates": meta_res, "avg": sum(meta_res)/len(meta_res)}

# SpaceInvaders
space_res = []
for log in sorted(glob.glob("spaceinvaders.log")):
    task, rate = parse_log(log)
    if task is not None:
        space_res.append(rate)
results["SpaceInvaders"] = {"success_rates": space_res, "avg": sum(space_res)/len(space_res)}

# Freeway
freeway_res = []
for log in sorted(glob.glob("freeway.log")):
    task, rate = parse_log(log)
    if task is not None:
        freeway_res.append(rate)
results["Freeway"] = {"success_rates": freeway_res, "avg": sum(freeway_res)/len(freeway_res)}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results written to results.json")
PY

echo "=== Done ==="