import sys
import os
import argparse
import numpy as np
import torch
from src.toxic_vectors import ToxicVectorExtractor
from src.dpo_simulation import DPOSimulation
from src.unalign import Unaligner
from src.generate_results import generate_results

def main():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Starting reproduction...")
    
    # Step 1: Extract toxic vectors
    print("Step 1: Extracting toxic vectors...")
    extractor = ToxicVectorExtractor()
    extractor.train_probe(args.data, os.path.join(args.output, 'probe_model.pkl'))
    vectors, svd_vectors = extractor.extract_toxic_vectors()
    np.save(os.path.join(args.output, 'toxic_vectors.npy'), vectors)
    np.save(os.path.join(args.output, 'svd_vectors.npy'), svd_vectors)
    
    # Step 2: Simulate DPO
    print("Step 2: Simulating DPO alignment...")
    dpo = DPOSimulation(args.model)
    dpo.load_model()
    dpo.apply_dpo(os.path.join(args.output, 'toxic_vectors.npy'), 
                   os.path.join(args.output, 'svd_vectors.npy'), 
                   os.path.join(args.output, 'dpo_model_weights.pth'))
    
    # Step 3: Unalign
    print("Step 3: Undoing alignment...")
    unaligner = Unaligner(os.path.join(args.output, 'dpo_model_weights.pth'), 
                         os.path.join(args.output, 'toxic_vectors.npy'))
    unaligner.load_weights()
    unaligner.unalign(num_vectors=7, scale_factor=10, 
                     output_path=os.path.join(args.output, 'unaligned_model.pth'))
    
    # Step 4: Generate results
    print("Step 4: Generating results summary...")
    generate_results(args.output, os.path.join(args.output, 'results.json'))
    
    print("Reproduction completed successfully!")

if __name__ == "__main__":
    main()