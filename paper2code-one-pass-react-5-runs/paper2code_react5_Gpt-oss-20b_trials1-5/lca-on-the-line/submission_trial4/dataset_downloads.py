#!/usr/bin/env python3
"""
Utility script to download small subsets of ImageNet and its
out‑of‑distribution variants for quick experimentation.

The full datasets are large and cannot be stored in the repo.
This script downloads a 1 k‑image subset for each split
(available from the robustbench project) and extracts them
into the `data/` directory.

Run with:
    python dataset_downloads.py
"""

import os
import sys
import tarfile
import urllib.request

URLS = {
    "imagenet_val": (
        "https://github.com/pytorch/hub/raw/master/imagenet_val.tar",
        "data/imagenet/imagenet_val.tar",
    ),
    "imagenet_sketch": (
        "https://github.com/robustbench/robustbench/raw/master/data/imagenet_sketch_val.tar",
        "data/imagenet_sketch/imagenet_sketch_val.tar",
    ),
    "imagenet_render": (
        "https://github.com/robustbench/robustbench/raw/master/data/imagenet_render_val.tar",
        "data/imagenet_render/imagenet_render_val.tar",
    ),
    "imagenet_adversarial": (
        "https://github.com/robustbench/robustbench/raw/master/data/imagenet_adv_val.tar",
        "data/imagenet_adv/imagenet_adv_val.tar",
    ),
    "objectnet": (
        "https://github.com/robustbench/robustbench/raw/master/data/objectnet_val.tar",
        "data/objectnet/objectnet_val.tar",
    ),
}

def download_and_extract(name, url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isfile(dst):
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url, dst)
    else:
        print(f"{name} already downloaded.")
    print(f"Extracting {dst} ...")
    with tarfile.open(dst, "r:gz") as tar:
        tar.extractall(path=os.path.dirname(dst))

def main():
    for name, (url, dst) in URLS.items():
        download_and_extract(name, url, dst)

if __name__ == "__main__":
    main()