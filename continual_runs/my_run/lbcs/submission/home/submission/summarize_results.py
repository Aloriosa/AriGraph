import os
import json
import csv
import argparse
from collections import defaultdict

def summarize_results(input_dir, output_file):
    """Summarize results from all dataset runs"""
    results = []
    
    # Walk through all subdirectories in input_dir
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        results_json_path = os.path.join(subdir_path, 'results.json')
        if not os.path.exists(results_json_path):
            continue
            
        # Load results
        with open(results_json_path, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    # Write summary to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'dataset', 'coreset_size', 'validation_accuracy', 
            'test_accuracy', 'baseline_accuracy', 'performance_constraint',
            'seed', 'coreset_size_requested', 'performance_threshold'
        ])
        
        # Write data
        for result in results:
            writer.writerow([
                result['dataset'],
                result['coreset_size'],
                result['validation_accuracy'],
                result['test_accuracy'],
                result['baseline_accuracy'],
                result['performance_constraint'],
                result['seed'],
                result['coreset_size_requested'],
                result['performance_threshold']
            ])
    
    print(f"Summary written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Summarize LBCS results')
    parser.add_argument('--input_dir', type=str, default='results',
                        help='Directory containing result subdirectories')
    parser.add_argument('--output', type=str, default='results/summary.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    summarize_results(args.input_dir, args.output)

if __name__ == '__main__':
    main()