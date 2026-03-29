#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Capture OpenRouter credits before paperbench (JSON on stdout).
before_json="$(python3 openrouter.py)"

python3 airi_check.py

after_json="$(python3 openrouter.py)"

# Print per-key difference: second_run - first_run (numeric leaves); non-numeric / structural mismatches shown as before/after.
OPENROUTER_BEFORE_JSON="$before_json" OPENROUTER_AFTER_JSON="$after_json" python3 - <<'PY'
import json
import os


def diff(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out[k] = {"only_after": b[k]}
            elif k not in b:
                out[k] = {"only_before": a[k]}
            else:
                out[k] = diff(a[k], b[k])
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return b - a
    return {"before": a, "after": b}


before = json.loads(os.environ["OPENROUTER_BEFORE_JSON"])
after = json.loads(os.environ["OPENROUTER_AFTER_JSON"])
print(json.dumps(diff(before, after), indent=2))
PY
