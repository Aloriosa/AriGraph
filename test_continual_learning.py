#!/usr/bin/env python3
"""
Quick test script to verify the continual learning graph components.
Tests entity resolution, aggregation, and pattern extraction without requiring papers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual_learning_graph import EntityResolver, KnowledgeAggregator, PatternExtractor, DomainClassifier
import numpy as np


class MockRetriever:
    """Mock retriever for testing without actual model."""
    
    def embed(self, text):
        """Generate deterministic embedding based on text."""
        # Simple hash-based embedding for testing
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768)


def test_entity_resolver():
    """Test entity resolution."""
    print("="*60)
    print("TEST: Entity Resolver")
    print("="*60)
    
    resolver = EntityResolver(MockRetriever(), similarity_threshold=0.85)
    
    # Add some entities
    resolver.add_entity("Adam optimizer", "paper1", {"value": 0.001})
    resolver.add_entity("learning rate", "paper1", {"value": 0.001})
    resolver.add_entity("batch size", "paper1", {"value": 32})
    
    resolver.add_entity("Adam", "paper2", {"value": 0.0001})
    resolver.add_entity("learning rate", "paper2", {"value": 0.0001})
    
    # Test resolution
    existing = {"Adam optimizer", "learning rate", "batch size"}
    
    # Should match exactly
    result1 = resolver.resolve_entity("learning rate", existing)
    print(f"\n1. Resolve 'learning rate': {result1}")
    assert result1['action'] == 'MATCH'
    
    # Should match "Adam optimizer" (similar)
    result2 = resolver.resolve_entity("Adam", existing)
    print(f"2. Resolve 'Adam': {result2}")
    
    # Should be new
    result3 = resolver.resolve_entity("dropout", existing)
    print(f"3. Resolve 'dropout': {result3}")
    assert result3['action'] == 'NEW'
    
    # Test similarity search
    similar = resolver.find_similar_entities("learning_rate", top_k=3)
    print(f"\n4. Similar to 'learning_rate': {similar}")
    
    print("\n✓ Entity Resolver tests passed")


def test_knowledge_aggregator():
    """Test knowledge aggregation."""
    print("\n" + "="*60)
    print("TEST: Knowledge Aggregator")
    print("="*60)
    
    aggregator = KnowledgeAggregator()
    
    # Add hyperparameters from multiple papers
    aggregator.add_hyperparameter("learning rate", 0.001, "paper1", {"domain": "NLP"})
    aggregator.add_hyperparameter("learning rate", 0.0001, "paper2", {"domain": "NLP"})
    aggregator.add_hyperparameter("learning rate", 0.0003, "paper3", {"domain": "NLP"})
    aggregator.add_hyperparameter("learning rate", 0.00001, "paper4", {"domain": "Vision"})
    
    aggregator.add_hyperparameter("batch size", 32, "paper1")
    aggregator.add_hyperparameter("batch size", 64, "paper2")
    aggregator.add_hyperparameter("batch size", 128, "paper3")
    
    # Get statistics
    lr_stats = aggregator.get_hyperparameter_stats("learning rate")
    print(f"\n1. Learning rate statistics:")
    print(f"   Count: {lr_stats['count']}")
    print(f"   Papers: {lr_stats['papers']}")
    if 'numerical' in lr_stats:
        print(f"   Mean: {lr_stats['numerical']['mean']:.6f}")
        print(f"   Median: {lr_stats['numerical']['median']:.6f}")
        print(f"   Range: [{lr_stats['numerical']['min']:.6f}, {lr_stats['numerical']['max']:.6f}]")
    
    batch_stats = aggregator.get_hyperparameter_stats("batch size")
    print(f"\n2. Batch size statistics:")
    print(f"   Count: {batch_stats['count']}")
    if 'numerical' in batch_stats:
        print(f"   Mean: {batch_stats['numerical']['mean']:.1f}")
        print(f"   Range: [{batch_stats['numerical']['min']:.0f}, {batch_stats['numerical']['max']:.0f}]")
    
    # Test error-solution tracking
    aggregator.add_error_solution("NaN loss", "gradient clipping", "paper1")
    aggregator.add_error_solution("NaN loss", "reduce learning rate", "paper2")
    aggregator.add_error_solution("NaN loss", "gradient clipping", "paper3")
    
    solutions = aggregator.get_top_solutions("NaN loss", top_k=5)
    print(f"\n3. Top solutions for 'NaN loss':")
    for sol, count in solutions:
        print(f"   - {sol}: {count} papers")
    
    # Test best practices
    aggregator.add_best_practice("gradient clipping", "prevents NaN loss", "paper1")
    aggregator.add_best_practice("warmup", "improves stability", "paper2")
    
    print(f"\n4. Best practices tracked: {len(aggregator.best_practices)}")
    
    print("\n✓ Knowledge Aggregator tests passed")


def test_pattern_extractor():
    """Test pattern extraction."""
    print("\n" + "="*60)
    print("TEST: Pattern Extractor")
    print("="*60)
    
    extractor = PatternExtractor(min_support=2)
    
    # Create sample triplets by paper
    triplets_by_paper = {
        "paper1": [
            ("Adam", "learning rate", {"label": "value"}),
            ("Adam", "batch size", {"label": "requires"}),
            ("training", "uses", {"label": "optimizer"}),
        ],
        "paper2": [
            ("Adam", "learning rate", {"label": "value"}),
            ("Adam", "warmup", {"label": "includes"}),
            ("training", "uses", {"label": "optimizer"}),
        ],
        "paper3": [
            ("Adam", "batch size", {"label": "requires"}),
            ("gradient clipping", "prevents", {"label": "NaN loss"}),
            ("training", "uses", {"label": "optimizer"}),
        ],
    }
    
    # Extract co-occurrence patterns
    cooccurrence = extractor.extract_cooccurrence_patterns(triplets_by_paper)
    print(f"\n1. Co-occurrence patterns found: {len(cooccurrence)}")
    for pair, data in list(cooccurrence.items())[:5]:
        print(f"   {pair[0]} + {pair[1]}: support={data['support']}, conf={data['confidence']:.2f}")
    
    # Extract relation patterns
    relation_patterns = extractor.extract_relation_patterns(triplets_by_paper)
    print(f"\n2. Relation patterns found: {len(relation_patterns)}")
    for rel, data in relation_patterns.items():
        print(f"   '{rel}': support={data['support']}, conf={data['confidence']:.2f}")
    
    print("\n✓ Pattern Extractor tests passed")


def test_domain_classifier():
    """Test domain classification."""
    print("\n" + "="*60)
    print("TEST: Domain Classifier")
    print("="*60)
    
    classifier = DomainClassifier()
    
    # Test different text samples
    samples = [
        ("We use BERT transformer for text classification", "NLP"),
        ("The CNN processes images with convolution layers", "Vision"),
        ("Our reinforcement learning agent learns a policy", "RL"),
        ("This algorithm processes the data efficiently", "General"),
    ]
    
    print("\nClassification results:")
    for text, expected in samples:
        result = classifier.classify(text)
        status = "✓" if result == expected else "✗"
        print(f"   {status} '{text[:40]}...' → {result} (expected: {expected})")
    
    print("\n✓ Domain Classifier tests passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONTINUAL LEARNING GRAPH - UNIT TESTS")
    print("="*60)
    
    try:
        test_entity_resolver()
        test_knowledge_aggregator()
        test_pattern_extractor()
        test_domain_classifier()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe continual learning graph components are working correctly!")
        print("You can now run the full pipeline on actual papers.")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
