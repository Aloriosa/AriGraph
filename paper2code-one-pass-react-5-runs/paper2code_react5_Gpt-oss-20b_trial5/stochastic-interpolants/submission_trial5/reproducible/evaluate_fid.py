import argparse
import yaml
import torch
import os
from torch_fidelity import calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Assume we have a reference folder of real images:
    # For a real evaluation you need a large set of real images.
    # Here we just point to the dataset folder.
    real_dir = cfg['dataset_path']
    gen_dir = 'samples'

    metrics = calculate_metrics(
        input1_paths=gen_dir,
        input2_paths=real_dir,
        cuda=True,
        batch_size=8,
        output_dir='metrics',
        verbose=True
    )
    print("FID:", metrics['fid'])

if __name__ == "__main__":
    main()