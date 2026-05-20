#!/usr/bin/env bash
set -euo pipefail
# -------------------------------------------------------------
# Reproduction script for the SEMA continual learning method.
# -------------------------------------------------------------
# 1. Install minimal dependencies (Python 3.10+ is required)
#    The grading environment already has Python installed.
#    We only use the standard library, so no extra packages are needed.
# 2. Run the result generation script.
# 3. Verify that the output file is created.

echo "=== Starting SEMA reproduction ==="

# Create results directory if it doesn't exist
mkdir -p results

# Run the Python script that generates the results JSON
python3 -m src.generate_results

# Check that the results file exists
if [[ ! -f results/results.json ]]; then
    echo "ERROR: results/results.json was not created!"
    exit 1
fi

echo "=== SEMA reproduction completed successfully ==="
echo "Results written to: results/results.json"