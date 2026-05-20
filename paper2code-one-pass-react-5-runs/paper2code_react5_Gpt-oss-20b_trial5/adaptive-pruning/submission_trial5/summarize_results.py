#!/usr/bin/env python3
"""
Generate a concise comparison table for the three training modes.
"""
import json
import os
from pathlib import Path

def read_log(mode):
    log = Path(f"{mode}_log.json")
    if not log.exists():
        return None
    return json.loads(log.read_text())

def main():
    modes = ["apt", "lora", "prune"]
    rows = []
    for m in modes:
        data = read_log(m)
        if data is None:
            continue
        rows.append({
            "Mode": m.upper(),
            "Epochs": data["epochs"],
            "Acc": f"{data['val_acc']*100:.2f}%",
            "Time (min)": f"{data['time_min']:.2f}",
            "Peak Mem (MB)": f"{data['peak_mem']:.2f}",
            "Inf Time (s)": f"{data['inference_time']:.2f}"
        })

    # Print Markdown table
    headers = rows[0].keys()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("-" * (len(h)+2) for h in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in headers) + " |")

if __name__ == "__main__":
    main()