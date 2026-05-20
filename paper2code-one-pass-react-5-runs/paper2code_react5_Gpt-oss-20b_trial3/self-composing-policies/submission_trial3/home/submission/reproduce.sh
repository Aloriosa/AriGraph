#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1. Install dependencies (will use the system python, no venv)
# ------------------------------------------------------------------
pip install -q -r requirements.txt

# ------------------------------------------------------------------
# 2. Create output directory
# ------------------------------------------------------------------
mkdir -p results
cd results

# ------------------------------------------------------------------
# 3. Run Meta‑World experiment
# ------------------------------------------------------------------
echo "=== Meta‑World ==="
python ../experiments/meta_world.py > meta_world.log 2>&1
echo "Meta‑World finished"

# ------------------------------------------------------------------
# 4. Run Atari experiment
# ------------------------------------------------------------------
echo "=== Atari (SpaceInvaders + Freeway) ==="
python ../experiments/ale.py > ale.log 2>&1
echo "Atari finished"

# ------------------------------------------------------------------
# 5. Merge CSVs
# ------------------------------------------------------------------
cat meta_world_results.csv ale_results.csv > ../results.csv
# Clean temporary CSVs
rm meta_world_results.csv ale_results.csv

echo "=== Reproduction complete ==="