import os
import re
import json

def parse_log(log_path):
    accuracy = None
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r"Test Accuracy.*: ([0-9.]+)%", line)
            if m:
                accuracy = float(m.group(1))
    return accuracy

results = {}
for root, dirs, files in os.walk("checkpoints"):
    for d in dirs:
        dirpath = os.path.join(root, d)
        log_path = os.path.join(dirpath, "train.log")
        if os.path.exists(log_path):
            acc = parse_log(log_path)
            if acc is not None:
                results[d] = acc

print("Test Accuracies:")
for k, v in sorted(results.items()):
    print(f"{k:20s}: {v:.2f}%")