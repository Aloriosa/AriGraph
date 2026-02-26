#!/usr/bin/env python3
"""
Example script demonstrating how to use the Continual Learning Graph.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual_learning_graph import ContinualLearningGraph
from utils.utils import Logger


def example_usage():
    """
    Example: Process multiple papers and query the unified knowledge graph.
    """
    
    # Setup
    log_path = "example_continual_learning"
    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path)
    
    # Initialize
    base_url = "https://inference.airi.net:46783/v1"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3Njk0NDQ0MjQsImV4cCI6MTc3MDA0OTIyNH0.4eyt0_zsvfDY4HmwDsc4eS0p0mFDftFQL7u_DhRJqt4"
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
    
    log("="*70)
    log("CONTINUAL LEARNING GRAPH EXAMPLE")
    log("="*70)
    
    # Create continual learning graph
    cl_graph = ContinualLearningGraph(
        model=model,
        system_prompt="You are a helpful assistant specializing in research reproduction",
        api_key=api_key,
        log=log,
        base_url=base_url,
        device="cpu"
    )
    
    # Example: Process multiple papers
    papers = [
        {
            'paper_path': '/path/to/paper1/paper.md',
            'repo_url': 'https://github.com/org/repo1'
        },
        {
            'paper_path': '/path/to/paper2/paper.md',
            'repo_url': 'https://github.com/org/repo2'
        }
    ]
    
    log("\n" + "="*70)
    log("PROCESSING PAPERS")
    log("="*70)
    
    for i, paper in enumerate(papers, 1):
        log(f"\n>>> Processing paper {i}/{len(papers)}...")
        try:
            summary = cl_graph.process_paper(
                paper['paper_path'],
                paper['repo_url'],
                max_code_files=10  # Limit code files for speed
            )
            log(f"✓ Paper {i} processed: {summary['triplets_added']} triplets added")
        except Exception as e:
            log(f"✗ Error processing paper {i}: {str(e)}")
    
    # Query unified knowledge
    log("\n" + "="*70)
    log("QUERYING UNIFIED KNOWLEDGE")
    log("="*70)
    
    # Query 1: Get meta-knowledge about a concept
    log("\n>>> Query: What do we know about 'learning rate' across all papers?")
    lr_knowledge = cl_graph.get_meta_knowledge("learning rate")
    log(f"Found in {len(lr_knowledge['papers'])} papers")
    log(f"Total mentions: {lr_knowledge['frequency']}")
    
    if lr_knowledge.get('hyperparameter_stats'):
        stats = lr_knowledge['hyperparameter_stats']
        if 'numerical' in stats:
            log(f"Numerical statistics:")
            log(f"  Mean: {stats['numerical']['mean']:.6f}")
            log(f"  Median: {stats['numerical']['median']:.6f}")
            log(f"  Range: [{stats['numerical']['min']:.6f}, {stats['numerical']['max']:.6f}]")
    
    # Query 2: Get solutions for an error
    log("\n>>> Query: How to fix 'NaN loss'?")
    solutions = cl_graph.query_errors("NaN loss")
    log(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions[:5], 1):
        log(f"  {i}. {sol['solution']} (confidence: {sol['confidence']:.2f})")
    
    # Query 3: Get frequent patterns
    log("\n>>> Query: What are the most common patterns?")
    if 'patterns' in cl_graph.meta_graph:
        patterns = cl_graph.meta_graph['patterns']
        
        if 'cooccurrence' in patterns:
            log(f"\nFrequent entity co-occurrences:")
            for pair, data in list(patterns['cooccurrence'].items())[:10]:
                log(f"  {pair[0]} + {pair[1]}: appears in {data['support']} papers")
        
        if 'relations' in patterns:
            log(f"\nMost common relations:")
            sorted_rels = sorted(
                patterns['relations'].items(),
                key=lambda x: x[1]['support'],
                reverse=True
            )
            for rel, data in sorted_rels[:10]:
                log(f"  {rel}: {data['support']} papers (conf: {data['confidence']:.2f})")
    
    # Query 4: Entity statistics
    log("\n>>> Query: What are the most frequently mentioned entities?")
    top_entities = cl_graph.aggregator.entity_frequencies.most_common(15)
    for entity, count in top_entities:
        papers = len(cl_graph.entity_resolver.entity_metadata.get(entity, {}).get('papers', []))
        log(f"  {entity}: {count} mentions across {papers} papers")
    
    # Save checkpoint
    checkpoint_path = os.path.join(log_path, "checkpoint.pkl")
    log(f"\n>>> Saving checkpoint to {checkpoint_path}...")
    cl_graph.save_checkpoint(checkpoint_path)
    log("✓ Checkpoint saved")
    
    # Demonstrate loading
    log(f"\n>>> Loading checkpoint from {checkpoint_path}...")
    new_graph = ContinualLearningGraph(
        model=model,
        system_prompt="You are a helpful assistant specializing in research reproduction",
        api_key=api_key,
        log=log,
        base_url=base_url,
        device="cpu"
    )
    new_graph.load_checkpoint(checkpoint_path)
    log(f"✓ Checkpoint loaded: {new_graph.paper_count} papers")
    
    log("\n" + "="*70)
    log("EXAMPLE COMPLETE")
    log("="*70)
    log(f"Total papers processed: {cl_graph.paper_count}")
    log(f"Total triplets: {len(cl_graph.triplets)}")
    log(f"Total entities: {len(cl_graph.aggregator.entity_frequencies)}")
    log(f"Total cost: ${cl_graph.total_amount:.4f}")


if __name__ == "__main__":
    # For actual usage, provide real paper paths
    print("="*70)
    print("CONTINUAL LEARNING GRAPH - EXAMPLE USAGE")
    print("="*70)
    print("\nThis is an example script showing how to use the continual learning graph.")
    print("\nTo run with real data, use the main script:")
    print("\n  python continual_learning_graph.py \\")
    print("    --papers paper1 paper2 paper3 \\")
    print("    --paper-base-path /path/to/papers \\")
    print("    --log-path output \\")
    print("    --save-checkpoint checkpoint.pkl")
    print("\nTo query after processing:")
    print("\n  >>> from continual_learning_graph import ContinualLearningGraph")
    print("  >>> graph = ContinualLearningGraph(...)")
    print("  >>> graph.load_checkpoint('checkpoint.pkl')")
    print("  >>> knowledge = graph.get_meta_knowledge('learning rate')")
    print("  >>> solutions = graph.query_errors('NaN loss')")
    print("\n" + "="*70)
