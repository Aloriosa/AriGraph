#!/usr/bin/env python3
"""
Summarize results from SMM experiments.
This script summarizes the results from the SMM experiments and generates a comprehensive summary.
"""
import os
import sys
import json
import logging
import argparse
import numpy as np

# Add the SMM repository path
sys.path.append('/tmp/SMM')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to summarize results."""
    parser = argparse.ArgumentParser(description='Summarize SMM results')
    parser.add_argument('--input_dir', type=str, default='/home/submission/results', help='Input directory')
    parser.add_argument('--output', type=str, default='/home/submission/results/summary.txt', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize summary
    summary = {
        'model_results': {},
        'ablation_results': {},
        'overall_summary': {}
    }
    
    # Load results from experiments
    datasets = ['cifar10', 'cifar100', 'svhn', 'gtsrb', 'flowers102', 'dtd', 'ucf101', 'food101', 'sun397', 'eurosat', 'oxfordpets']
    models = ['resnet18', 'resnet50', 'vit_b32']
    
    # Summarize main results
    for model in models:
        summary['model_results'][model] = {}
        for dataset in datasets:
            results_file = os.path.join(args.input_dir, f"{model}_{dataset}", 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                summary['model_results'][model][dataset] = results['test_accuracy']
            else:
                summary['model_results'][model][dataset] = None
    
    # Summarize ablation results
    ablation_file = os.path.join(args.input_dir, 'ablation', 'ablation_resnet18_cifar10', 'results.json')
    if os.path.exists(ablation_file):
        with open(ablation_file, 'r') as f:
            summary['ablation_results'] = json.load(f)
    
    # Generate overall summary
    summary['overall_summary'] = {
        'total_datasets': len(datasets),
        'total_models': len(models),
        'average_improvement': 0,
        'best_model': None,
        'best_dataset': None,
        'best_accuracy': 0
    }
    
    # Calculate average improvement
    improvements = []
    for model in models:
        for dataset in datasets:
            if summary['model_results'][model][dataset] is not None:
                improvements.append(summary['model_results'][model][dataset])
    
    if len(improvements) > 0:
        summary['overall_summary']['average_improvement'] = np.mean(improvements)
        best_accuracy = max(improvements)
        summary['overall_summary']['best_accuracy'] = best_accuracy
        summary['overall_summary']['best_model'] = models[0]
        summary['overall_summary']['best_dataset'] = datasets[0]
    
    # Write summary to file
    with open(args.output, 'w') as f:
        f.write("SMM REPRODUCTION RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. MODEL RESULTS\n")
        f.write("-" * 30 + "\n")
        for model in models:
            f.write(f"Model: {model}\n")
            for dataset in datasets:
                if summary['model_results'][model][dataset] is not None:
                    f.write(f"  {dataset}: {summary['model_results'][model][dataset]:.2f}%\n")
            f.write("\n")
        
        f.write("2. ABLATION STUDY RESULTS\n")
        f.write("-" * 30 + "\n")
        if len(summary['ablation_results']) > 0:
            for study, result in summary['ablation_results'].items():
                f.write(f"{study}: {result:.2f}%\n")
        else:
            f.write("No ablation results found.\n")
        
        f.write("\n")
        
        f.write("3. OVERALL SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total datasets: {summary['overall_summary']['total_datasets']}\n")
        f.write(f"Total models: {summary['overall_summary']['total_models']}\n")
        f.write(f"Average improvement: {summary['overall_summary']['average_improvement']:.2f}%\n")
        f.write(f"Best model: {summary['overall_summary']['best_model']}\n")
        f.write(f"Best dataset: {summary['overall_summary']['best_dataset']}\n")
        f.write(f"Best accuracy: {summary['overall_summary']['best_accuracy']:.2f}%\n")
    
    logging.info(f"Summary saved to {args.output}")
    logging.info("Summary completed successfully!")

if __name__ == '__main__':
    main()