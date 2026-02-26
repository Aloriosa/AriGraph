# Continual Learning Knowledge Graph for Paper Reproduction

A system that processes multiple research papers and their code repositories to build a unified, generalizing knowledge base. The graph consolidates common approaches, tools, and frameworks across papers to become a source of unified knowledge on research paper reproduction.

## Overview

The continual learning graph extends the single-paper reproduction graph to:
- **Consolidate entities** across papers (e.g., recognize "Adam optimizer" as the same concept across 10 papers)
- **Aggregate statistics** for hyperparameters (e.g., learning rate ranges: 0.0001-0.001 across 20 papers)
- **Extract patterns** that appear frequently (e.g., "warmup + Adam + gradient clipping" in 15 papers)
- **Track solutions** to common errors (e.g., "NaN loss → gradient clipping" from 8 papers)
- **Build meta-knowledge** that generalizes beyond individual papers

## Key Features

### 1. Entity Resolution
- Automatically matches entities across papers using embedding similarity
- Identifies variants (e.g., "BERT", "RoBERTa", "DistilBERT" → variants of Transformer)
- Maintains canonical forms to avoid duplicates

### 2. Knowledge Aggregation
- **Hyperparameters**: Tracks distributions of values across papers
  - Mean, median, std, range
  - Context-specific (by domain, architecture, dataset)
- **Errors & Solutions**: Accumulates troubleshooting knowledge
- **Best Practices**: Vote-weighted by paper frequency

### 3. Pattern Extraction
- **Co-occurrence patterns**: Entities that frequently appear together
- **Relation patterns**: Common relationship types
- **Templates**: Extracted training configurations

### 4. Multi-Level Querying
- Query general patterns: "How to train a Transformer?"
- Query domain-specific: "Learning rate for NLP?"
- Query troubleshooting: "How to fix NaN loss?"
- Query statistics: "What's the typical batch size?"

## Installation

Requires the existing arigraph dependencies:
```bash
# Your existing environment
pip install networkx pyvis numpy
```

## Usage

### Basic Usage: Process Multiple Papers

```bash
python continual_learning_graph.py \
  --papers adaptive-pruning memory-efficient-training attention-variants \
  --paper-base-path /path/to/papers \
  --log-path output \
  --save-checkpoint graph_checkpoint.pkl \
  --max-code-files 20
```

### Loading and Querying

```python
from continual_learning_graph import ContinualLearningGraph
from utils.utils import Logger

# Initialize
log = Logger("query_output")
graph = ContinualLearningGraph(model, system_prompt, api_key, log, base_url)

# Load existing checkpoint
graph.load_checkpoint("graph_checkpoint.pkl")

# Query 1: Get meta-knowledge about a concept
knowledge = graph.get_meta_knowledge("learning rate")
print(f"Found in {len(knowledge['papers'])} papers")
print(f"Mean: {knowledge['hyperparameter_stats']['numerical']['mean']}")
print(f"Range: {knowledge['hyperparameter_stats']['numerical']['range']}")

# Query 2: Get solutions for errors
solutions = graph.query_errors("NaN loss")
for sol in solutions:
    print(f"{sol['solution']} (confidence: {sol['confidence']:.2f})")

# Query 3: Get entity frequencies
top_entities = graph.aggregator.entity_frequencies.most_common(20)
for entity, count in top_entities:
    papers = len(graph.entity_resolver.entity_metadata[entity]['papers'])
    print(f"{entity}: {count} mentions across {papers} papers")

# Query 4: Get patterns
patterns = graph.meta_graph['patterns']
for pair, data in patterns['cooccurrence'].items():
    print(f"{pair[0]} + {pair[1]}: {data['support']} papers")
```

### Incremental Processing

```python
# Start fresh
graph = ContinualLearningGraph(...)

# Process first batch of papers
for paper in papers_batch_1:
    graph.process_paper(paper['path'], paper['repo_url'])

# Save checkpoint
graph.save_checkpoint("checkpoint_batch1.pkl")

# Later: Load and add more papers
graph.load_checkpoint("checkpoint_batch1.pkl")

# Process second batch
for paper in papers_batch_2:
    graph.process_paper(paper['path'], paper['repo_url'])

# Save updated checkpoint
graph.save_checkpoint("checkpoint_batch2.pkl")
```

## Architecture

```
continual_learning_graph.py
├── EntityResolver
│   ├── Embedding-based similarity matching
│   ├── Canonical form mapping
│   └── Entity metadata tracking
├── KnowledgeAggregator
│   ├── Hyperparameter statistics
│   ├── Error-solution tracking
│   └── Best practices accumulation
├── PatternExtractor
│   ├── Co-occurrence mining
│   ├── Relation pattern detection
│   └── Template extraction
└── ContinualLearningGraph
    ├── Paper processing pipeline
    ├── Entity resolution
    ├── Knowledge consolidation
    └── Multi-level querying
```

## Processing Pipeline

For each paper:
1. **Extract**: Extract triplets from paper text + code
2. **Resolve**: Match entities to existing concepts
3. **Consolidate**: Update statistics and frequencies
4. **Aggregate**: Track hyperparameters, errors, practices
5. **Extract Patterns**: Mine frequent patterns (periodic)

## Output

### Checkpoint File (`checkpoint.pkl`)
Contains:
- All triplets from all papers
- Entity metadata and frequencies
- Aggregated statistics
- Extracted patterns
- Provenance information

### Summary JSON (`continual_learning_summary.json`)
Contains:
- Processing statistics
- Per-paper summaries
- Pattern summaries
- Cost tracking

### Log File (`log.txt`)
Detailed processing log with:
- Per-section progress
- Entity resolution stats
- Pattern extraction results

## Examples

### Example 1: Find typical hyperparameters for a domain

```python
# Process 20 NLP papers
for paper in nlp_papers:
    graph.process_paper(paper['path'], paper['repo_url'])

# Query learning rates
lr_stats = graph.get_meta_knowledge("learning rate")
print(f"Typical range: {lr_stats['hyperparameter_stats']['numerical']['range']}")
print(f"Median: {lr_stats['hyperparameter_stats']['numerical']['median']}")
```

### Example 2: Build a troubleshooting guide

```python
# After processing many papers
errors = ["NaN loss", "OOM error", "convergence failure", "shape mismatch"]

for error in errors:
    solutions = graph.query_errors(error)
    print(f"\n{error}:")
    for sol in solutions[:3]:
        print(f"  - {sol['solution']} ({sol['frequency']} papers)")
```

### Example 3: Extract common training templates

```python
# Get most frequent patterns
patterns = graph.meta_graph['patterns']['cooccurrence']

# Filter for training-related patterns
training_patterns = {
    k: v for k, v in patterns.items()
    if any(term in str(k).lower() for term in ['optimizer', 'learning', 'batch', 'training'])
}

print("Common training configurations:")
for pair, data in sorted(training_patterns.items(), key=lambda x: x[1]['support'], reverse=True)[:10]:
    print(f"{pair[0]} + {pair[1]}: {data['support']} papers")
```

## Command-Line Options

```
--papers PAPER1 PAPER2 ...        Paper names to process
--paper-base-path PATH            Base path to papers directory
--device cpu|cuda                 Device for retriever
--log-path PATH                   Output directory
--checkpoint PATH                 Load from checkpoint
--save-checkpoint PATH            Save checkpoint after processing
--max-code-files N                Max code files per paper (-1 for all)
```

## Key Differences from Single-Paper Graph

| Feature | Single-Paper | Continual Learning |
|---------|-------------|-------------------|
| Scope | One paper | Multiple papers |
| Entities | Paper-specific | Consolidated across papers |
| Values | Exact values | Statistical distributions |
| Knowledge | Specific | Generalized + Specific |
| Patterns | Not extracted | Automatically mined |
| Provenance | Implicit | Tracked per paper |

## Performance Considerations

- **Entity resolution**: O(n²) in entities, but cached with embeddings
- **Pattern extraction**: Runs every 5 papers to balance cost/benefit
- **Memory**: Checkpoint size grows with papers (~10-50MB per paper)
- **Cost**: Shares LLM calls with base graph, adds entity resolution overhead

## Future Enhancements

1. **Advanced pattern mining**: Subgraph isomorphism for complex patterns
2. **Domain-specific graphs**: Separate meta-graphs per domain
3. **Temporal tracking**: Track evolution of methods over time
4. **Conflict resolution**: Better handling of contradictory information
5. **Query optimization**: Index structures for faster retrieval
6. **Visualization**: Interactive graph exploration of meta-knowledge

## Related Files

- `test_paper_reproduction.py`: Single-paper graph (base implementation)
- `prompts/cookbook_extraction_prompt.py`: Extraction prompts
- `graphs/contriever_graph.py`: Base graph class
- `continual_learning_design.md`: Detailed design document

## Citation

If you use this system, please cite the original paper reproduction work and note the continual learning extension.
