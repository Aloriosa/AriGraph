#!/usr/bin/env python3
"""
Continual Learning Knowledge Graph for Paper Reproduction.
Builds a unified, generalizing knowledge base across multiple papers.
"""

import re
import os
import sys
import json
import pickle
import argparse
import subprocess
import numpy as np
from pathlib import Path
from time import time
from collections import Counter, defaultdict
import datetime
from typing import List, Dict, Tuple, Set, Optional, Any

import networkx as nx
from pyvis.network import Network

# Import existing modules
from graphs.contriever_graph import ContrieverGraph
from utils.utils import Logger, process_triplets

sys.path.insert(0, os.path.dirname(__file__))
from prompts.cookbook_extraction_prompt import prompt_cookbook_extraction
from test_paper_reproduction import (
    clone_repo, collect_code_files, read_file_safe, 
    extract_code_chunks, load_paper, split_into_sections,
    clean_latex, preprocess_section, vis_net
)


# =============================================================================
# ENTITY RESOLVER - Consolidates entities across papers
# =============================================================================

class EntityResolver:
    """
    Resolves and consolidates entities across multiple papers.
    Uses embedding similarity and co-occurrence patterns.
    """
    
    def __init__(self, retriever, similarity_threshold=0.85):
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.entity_embeddings = {}  # entity -> embedding
        self.entity_metadata = {}    # entity -> {papers, frequency, contexts}
        self.canonical_forms = {}    # variant -> canonical
        
    def add_entity(self, entity: str, paper_id: str, context: Dict = None):
        """Add or update entity information."""
        # Get or create embedding
        if entity not in self.entity_embeddings:
            self.entity_embeddings[entity] = self.retriever.embed(entity)
        
        # Update metadata
        if entity not in self.entity_metadata:
            self.entity_metadata[entity] = {
                'papers': set(),
                'frequency': 0,
                'contexts': []
            }
        
        self.entity_metadata[entity]['papers'].add(paper_id)
        self.entity_metadata[entity]['frequency'] += 1
        if context:
            self.entity_metadata[entity]['contexts'].append({
                'paper': paper_id,
                **context
            })
    
    def find_similar_entities(self, entity: str, top_k=5) -> List[Tuple[str, float]]:
        """Find similar entities using embedding similarity."""
        if not self.entity_embeddings:
            return []
        
        entity_emb = self.retriever.embed(entity)
        
        similarities = []
        for other_entity, other_emb in self.entity_embeddings.items():
            if other_entity == entity:
                continue
            
            # Cosine similarity
            similarity = np.dot(entity_emb, other_emb) / (
                np.linalg.norm(entity_emb) * np.linalg.norm(other_emb)
            )
            similarities.append((other_entity, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def resolve_entity(self, entity: str, existing_entities: Set[str]) -> Dict:
        """
        Resolve entity to canonical form.
        Returns: {
            'action': 'MATCH' | 'VARIANT' | 'NEW',
            'canonical': canonical entity name,
            'confidence': float
        }
        """
        # Exact match
        if entity in existing_entities:
            return {
                'action': 'MATCH',
                'canonical': entity,
                'confidence': 1.0
            }
        
        # Check if already mapped
        if entity in self.canonical_forms:
            return {
                'action': 'MATCH',
                'canonical': self.canonical_forms[entity],
                'confidence': 0.95
            }
        
        # Find similar entities
        similar = self.find_similar_entities(entity, top_k=1)
        
        if similar and similar[0][1] >= self.similarity_threshold:
            canonical = similar[0][0]
            confidence = similar[0][1]
            
            # Check if it's an exact variant (different casing, etc.)
            if entity.lower() == canonical.lower():
                self.canonical_forms[entity] = canonical
                return {
                    'action': 'MATCH',
                    'canonical': canonical,
                    'confidence': confidence
                }
            else:
                # It's a semantic variant
                return {
                    'action': 'VARIANT',
                    'canonical': canonical,
                    'confidence': confidence
                }
        
        # No match found - new entity
        return {
            'action': 'NEW',
            'canonical': entity,
            'confidence': 1.0
        }


# =============================================================================
# KNOWLEDGE AGGREGATOR - Statistical consolidation
# =============================================================================

class KnowledgeAggregator:
    """
    Aggregates knowledge across multiple papers.
    Computes statistics for hyperparameters, tracks frequencies, etc.
    """
    
    def __init__(self):
        self.hyperparameter_values = defaultdict(list)  # param -> [(value, paper, context)]
        self.entity_frequencies = Counter()
        self.relation_frequencies = Counter()
        self.error_solutions = defaultdict(list)  # error -> [(solution, papers)]
        self.best_practices = defaultdict(list)  # practice -> [(benefit, papers)]
        
    def add_hyperparameter(self, param: str, value: Any, paper_id: str, context: Dict = None):
        """Track hyperparameter value."""
        self.hyperparameter_values[param].append({
            'value': value,
            'paper': paper_id,
            'context': context or {}
        })
    
    def add_error_solution(self, error: str, solution: str, paper_id: str):
        """Track error-solution pairs."""
        self.error_solutions[error].append({
            'solution': solution,
            'paper': paper_id
        })
    
    def add_best_practice(self, practice: str, benefit: str, paper_id: str):
        """Track best practices."""
        self.best_practices[practice].append({
            'benefit': benefit,
            'paper': paper_id
        })
    
    def get_hyperparameter_stats(self, param: str) -> Dict:
        """Get statistics for a hyperparameter."""
        values = self.hyperparameter_values.get(param, [])
        if not values:
            return {}
        
        # Try to extract numerical values
        numerical_values = []
        for v in values:
            try:
                if isinstance(v['value'], (int, float)):
                    numerical_values.append(float(v['value']))
                else:
                    # Try to parse string
                    val_str = str(v['value'])
                    numerical_values.append(float(val_str))
            except (ValueError, TypeError):
                pass
        
        stats = {
            'count': len(values),
            'papers': list(set(v['paper'] for v in values)),
            'all_values': [v['value'] for v in values]
        }
        
        if numerical_values:
            stats['numerical'] = {
                'mean': np.mean(numerical_values),
                'median': np.median(numerical_values),
                'std': np.std(numerical_values),
                'min': np.min(numerical_values),
                'max': np.max(numerical_values),
                'range': [np.min(numerical_values), np.max(numerical_values)]
            }
        
        # Group by context (if available)
        context_groups = defaultdict(list)
        for v in values:
            ctx = v['context'].get('domain', 'general')
            context_groups[ctx].append(v['value'])
        
        if len(context_groups) > 1:
            stats['by_context'] = dict(context_groups)
        
        return stats
    
    def get_top_solutions(self, error: str, top_k=5) -> List[Tuple[str, int]]:
        """Get most frequent solutions for an error."""
        solutions = self.error_solutions.get(error, [])
        solution_counts = Counter(s['solution'] for s in solutions)
        return solution_counts.most_common(top_k)


# =============================================================================
# PATTERN EXTRACTOR - Mines frequent patterns
# =============================================================================

class PatternExtractor:
    """
    Extracts recurring patterns from accumulated knowledge.
    Finds common subgraphs and templates.
    """
    
    def __init__(self, min_support=3):
        self.min_support = min_support  # Minimum papers for a pattern
        self.patterns = {}  # pattern_id -> {subgraph, support, papers}
        
    def extract_cooccurrence_patterns(self, triplets_by_paper: Dict[str, List]) -> Dict:
        """Find entities that frequently co-occur."""
        # Track entity pairs across papers
        cooccurrences = defaultdict(set)  # (entity1, entity2) -> {papers}
        
        for paper_id, triplets in triplets_by_paper.items():
            entities_in_paper = set()
            for subj, obj, rel in triplets:
                entities_in_paper.add(subj)
                entities_in_paper.add(obj)
            
            # Record all pairs
            entities_list = list(entities_in_paper)
            for i, e1 in enumerate(entities_list):
                for e2 in entities_list[i+1:]:
                    pair = tuple(sorted([e1, e2]))
                    cooccurrences[pair].add(paper_id)
        
        # Filter by support
        frequent_patterns = {}
        for pair, papers in cooccurrences.items():
            if len(papers) >= self.min_support:
                frequent_patterns[pair] = {
                    'entities': pair,
                    'support': len(papers),
                    'papers': list(papers),
                    'confidence': len(papers) / len(triplets_by_paper)
                }
        
        return frequent_patterns
    
    def extract_relation_patterns(self, triplets_by_paper: Dict[str, List]) -> Dict:
        """Find common relation patterns (entity type -> relation -> entity type)."""
        relation_patterns = defaultdict(set)  # (rel_type) -> {papers}
        
        for paper_id, triplets in triplets_by_paper.items():
            for subj, obj, rel in triplets:
                rel_label = rel.get('label', 'unknown')
                relation_patterns[rel_label].add(paper_id)
        
        # Filter by support
        frequent_relations = {}
        for rel_label, papers in relation_patterns.items():
            if len(papers) >= self.min_support:
                frequent_relations[rel_label] = {
                    'relation': rel_label,
                    'support': len(papers),
                    'papers': list(papers),
                    'confidence': len(papers) / len(triplets_by_paper)
                }
        
        return frequent_relations


# =============================================================================
# CONTINUAL LEARNING GRAPH - Main class
# =============================================================================

class ContinualLearningGraph(ContrieverGraph):
    """
    Extended ContrieverGraph with continual learning capabilities.
    Consolidates knowledge across multiple papers.
    """
    
    def __init__(self, model, system_prompt, api_key, log, base_url='', device="cpu"):
        super().__init__(model, system_prompt, api_key, base_url, device)
        
        self.log = log
        self.reproduction_prompt = prompt_cookbook_extraction
        
        # Continual learning components
        self.entity_resolver = EntityResolver(self.retriever)
        self.aggregator = KnowledgeAggregator()
        self.pattern_extractor = PatternExtractor(min_support=2)
        
        # Tracking
        self.paper_count = 0
        self.papers_processed = []
        self.triplets_by_paper = {}  # paper_id -> triplets
        self.domain_classifier = DomainClassifier()
        
        # Meta-level knowledge
        self.meta_graph = {
            'patterns': {},
            'domain_specific': defaultdict(dict),
            'general_templates': {}
        }
        
        log(f'\nContinual Learning Graph initialized')
        log(f'Prompt:\n{self.reproduction_prompt}')
    
    def update_without_retrieve(self, observation, prev_subgraph, source_type="paper", paper_id=None):
        """Override to track paper-specific triplets."""
        example = [re.sub(r"Step \d+: ", "", triplet) for triplet in prev_subgraph]
        
        if source_type == "code":
            observation = f"[CODE IMPLEMENTATION]\n{observation}"
        else:
            observation = f"[PAPER TEXT]\n{observation}"
        
        prompt = self.reproduction_prompt.format(
            observation=observation, 
            example=example
        )
        
        response, _ = self.generate(prompt, t=0.001)
        
        # Process triplets
        new_triplets_raw = process_triplets(response)
        self.add_triplets(new_triplets_raw)
        
        new_triplets = self.exclude(new_triplets_raw)
        self.log(f"New {len(new_triplets)} triplets from {source_type}")
        
        # Track by paper
        if paper_id and paper_id not in self.triplets_by_paper:
            self.triplets_by_paper[paper_id] = []
        if paper_id:
            self.triplets_by_paper[paper_id].extend(new_triplets_raw)
        
        obs_embedding = self.retriever.embed(observation)
        new_triplets_str = self.convert(new_triplets_raw)
        obs_value = [new_triplets_str, obs_embedding]
        self.obs_episodic[observation] = obs_value
        
        return new_triplets_raw, obs_value
    
    def process_paper(self, paper_path: str, repo_url: str, max_code_files: int = -1) -> Dict:
        """
        Process a single paper with continual learning.
        """
        paper_id = Path(paper_path).parent.name
        self.log("\n" + "="*70)
        self.log(f"PROCESSING PAPER: {paper_id}")
        self.log("="*70)
        
        paper_start_time = time()
        
        # Track initial state
        initial_triplet_count = len(self.triplets)
        initial_entity_count = len(set(t[0] for t in self.triplets) | set(t[1] for t in self.triplets))
        
        # Initialize paper tracking
        self.triplets_by_paper[paper_id] = []
        
        # PHASE 1: Process paper text
        self.log("\nPHASE 1: Processing paper text...")
        paper_content = load_paper(paper_path)
        sections = split_into_sections(paper_content)
        self.log(f"Paper split into {len(sections)} sections")
        
        section_stats = []
        for i, section in enumerate(sections):
            section_start_time = time()
            self.log(f"\nSection {i+1}/{len(sections)}: {section['title']}")
            
            chunks = preprocess_section(section['content'])
            self.log(f"  {len(chunks)} chunks")
            
            for j, chunk in enumerate(chunks):
                try:
                    new_triplets, _ = self.update_without_retrieve(
                        chunk, [], source_type="paper", paper_id=paper_id
                    )
                except Exception as e:
                    self.log(f"  Error processing chunk {j+1}: {str(e)}")
                    continue
            
            section_time = time() - section_start_time
            section_stats.append({
                'section': section['title'],
                'time': section_time
            })
        
        # PHASE 2: Process code repository
        self.log("\nPHASE 2: Processing repository code...")
        temp_dir = f"temp_repos/{paper_id}_repo"
        
        try:
            if not os.path.exists(temp_dir):
                clone_repo(repo_url, temp_dir, self.log)
            
            code_files = collect_code_files(temp_dir, max_files=max_code_files)
            self.log(f"Processing {len(code_files)} code files")
            
            for idx, (rel_path, file_path) in enumerate(code_files):
                file_content = read_file_safe(file_path)
                
                if "[Binary file" in file_content or len(file_content) < 10:
                    continue
                
                chunks = extract_code_chunks(file_content, str(rel_path))
                
                for chunk in chunks:
                    try:
                        chunk_with_context = f"File: {rel_path}\n\n{chunk}"
                        self.update_without_retrieve(
                            chunk_with_context, [], source_type="code", paper_id=paper_id
                        )
                    except Exception as e:
                        continue
        
        except Exception as e:
            self.log(f"Error processing repository: {str(e)}")
        
        # PHASE 3: Entity resolution and consolidation
        self.log("\nPHASE 3: Entity resolution and consolidation...")
        new_entities = self._extract_entities_from_paper(paper_id)
        resolution_stats = self._resolve_entities(new_entities, paper_id)
        
        # PHASE 4: Knowledge aggregation
        self.log("\nPHASE 4: Knowledge aggregation...")
        self._aggregate_knowledge(paper_id)
        
        # Update paper count and metadata
        self.paper_count += 1
        self.papers_processed.append({
            'paper_id': paper_id,
            'paper_path': paper_path,
            'repo_url': repo_url,
            'timestamp': datetime.datetime.now().isoformat(),
            'triplets_added': len(self.triplets) - initial_triplet_count
        })
        
        # PHASE 5: Pattern extraction (periodic)
        if self.paper_count % 5 == 0:
            self.log("\nPHASE 5: Pattern extraction...")
            self._extract_patterns()
        
        paper_time = time() - paper_start_time
        
        # Summary
        final_triplet_count = len(self.triplets)
        final_entity_count = len(set(t[0] for t in self.triplets) | set(t[1] for t in self.triplets))
        
        summary = {
            'paper_id': paper_id,
            'processing_time': paper_time,
            'triplets_added': final_triplet_count - initial_triplet_count,
            'total_triplets': final_triplet_count,
            'entities_added': final_entity_count - initial_entity_count,
            'total_entities': final_entity_count,
            'resolution_stats': resolution_stats,
            'total_papers': self.paper_count,
            'cost': self.total_amount
        }
        
        self.log("\n" + "="*70)
        self.log(f"PAPER SUMMARY: {paper_id}")
        self.log("="*70)
        self.log(f"Triplets added: {summary['triplets_added']}")
        self.log(f"Total triplets: {summary['total_triplets']}")
        self.log(f"Entities added: {summary['entities_added']}")
        self.log(f"Total entities: {summary['total_entities']}")
        self.log(f"Entities matched: {resolution_stats['matched']}")
        self.log(f"Entities merged: {resolution_stats['variants']}")
        self.log(f"New entities: {resolution_stats['new']}")
        self.log(f"Processing time: {paper_time:.2f}s")
        self.log(f"Total cost: ${summary['cost']:.4f}")
        
        return summary
    
    def _extract_entities_from_paper(self, paper_id: str) -> Set[str]:
        """Extract all entities from paper's triplets."""
        entities = set()
        for triplet in self.triplets_by_paper.get(paper_id, []):
            entities.add(triplet[0])
            entities.add(triplet[1])
        return entities
    
    def _resolve_entities(self, new_entities: Set[str], paper_id: str) -> Dict:
        """Resolve entities to canonical forms."""
        existing_entities = set()
        for paper_triplets in self.triplets_by_paper.values():
            for triplet in paper_triplets:
                existing_entities.add(triplet[0])
                existing_entities.add(triplet[1])
        
        stats = {'matched': 0, 'variants': 0, 'new': 0}
        
        for entity in new_entities:
            resolution = self.entity_resolver.resolve_entity(entity, existing_entities)
            
            if resolution['action'] == 'MATCH':
                stats['matched'] += 1
            elif resolution['action'] == 'VARIANT':
                stats['variants'] += 1
            else:
                stats['new'] += 1
            
            # Add entity metadata
            self.entity_resolver.add_entity(entity, paper_id)
        
        return stats
    
    def _aggregate_knowledge(self, paper_id: str):
        """Aggregate knowledge from paper triplets."""
        for triplet in self.triplets_by_paper.get(paper_id, []):
            subj, obj, rel = triplet
            rel_label = rel.get('label', '').lower()
            
            # Update frequencies
            self.aggregator.entity_frequencies[subj] += 1
            self.aggregator.entity_frequencies[obj] += 1
            self.aggregator.relation_frequencies[rel_label] += 1
            
            # Detect hyperparameters
            if any(kw in rel_label for kw in ['value', 'set to', 'equals', 'configured']):
                self.aggregator.add_hyperparameter(subj, obj, paper_id)
            
            # Detect error-solution patterns
            if 'solved by' in rel_label or 'fixed by' in rel_label:
                self.aggregator.add_error_solution(subj, obj, paper_id)
            
            # Detect best practices
            if any(kw in rel_label for kw in ['improves', 'prevents', 'recommended']):
                self.aggregator.add_best_practice(subj, obj, paper_id)
    
    def _extract_patterns(self):
        """Extract patterns from accumulated knowledge."""
        self.log("  Extracting co-occurrence patterns...")
        cooccurrence = self.pattern_extractor.extract_cooccurrence_patterns(
            self.triplets_by_paper
        )
        self.log(f"  Found {len(cooccurrence)} frequent entity pairs")
        
        self.log("  Extracting relation patterns...")
        relation_patterns = self.pattern_extractor.extract_relation_patterns(
            self.triplets_by_paper
        )
        self.log(f"  Found {len(relation_patterns)} frequent relations")
        
        self.meta_graph['patterns'] = {
            'cooccurrence': cooccurrence,
            'relations': relation_patterns
        }
    
    def get_meta_knowledge(self, concept: str) -> Dict:
        """Get aggregated knowledge about a concept across all papers."""
        # Find all mentions
        mentions = []
        for paper_id, triplets in self.triplets_by_paper.items():
            for subj, obj, rel in triplets:
                if concept.lower() in subj.lower() or concept.lower() in obj.lower():
                    mentions.append({
                        'paper': paper_id,
                        'triplet': (subj, obj, rel)
                    })
        
        # Get metadata
        metadata = self.entity_resolver.entity_metadata.get(concept, {})
        
        # Get hyperparameter stats if applicable
        hyperparam_stats = self.aggregator.get_hyperparameter_stats(concept)
        
        return {
            'concept': concept,
            'frequency': len(mentions),
            'papers': list(metadata.get('papers', [])),
            'mentions': mentions[:10],  # Limit output
            'hyperparameter_stats': hyperparam_stats,
            'metadata': metadata
        }
    
    def query_errors(self, error_symptom: str) -> List[Dict]:
        """Query for solutions to an error."""
        solutions = self.aggregator.get_top_solutions(error_symptom, top_k=10)
        
        results = []
        for solution, count in solutions:
            results.append({
                'solution': solution,
                'frequency': count,
                'confidence': count / self.paper_count if self.paper_count > 0 else 0
            })
        
        return results
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save graph state."""
        checkpoint = {
            'version': '1.0',
            'paper_count': self.paper_count,
            'papers_processed': self.papers_processed,
            'triplets': self.triplets,
            'triplets_by_paper': self.triplets_by_paper,
            'entity_metadata': self.entity_resolver.entity_metadata,
            'canonical_forms': self.entity_resolver.canonical_forms,
            'aggregator_state': {
                'hyperparameter_values': dict(self.aggregator.hyperparameter_values),
                'entity_frequencies': dict(self.aggregator.entity_frequencies),
                'relation_frequencies': dict(self.aggregator.relation_frequencies),
                'error_solutions': dict(self.aggregator.error_solutions),
                'best_practices': dict(self.aggregator.best_practices)
            },
            'meta_graph': self.meta_graph,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_cost': self.total_amount
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.log(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load graph state."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.paper_count = checkpoint['paper_count']
        self.papers_processed = checkpoint['papers_processed']
        self.triplets = checkpoint['triplets']
        self.triplets_by_paper = checkpoint['triplets_by_paper']
        self.entity_resolver.entity_metadata = checkpoint['entity_metadata']
        self.entity_resolver.canonical_forms = checkpoint['canonical_forms']
        
        agg_state = checkpoint['aggregator_state']
        self.aggregator.hyperparameter_values = defaultdict(list, agg_state['hyperparameter_values'])
        self.aggregator.entity_frequencies = Counter(agg_state['entity_frequencies'])
        self.aggregator.relation_frequencies = Counter(agg_state['relation_frequencies'])
        self.aggregator.error_solutions = defaultdict(list, agg_state['error_solutions'])
        self.aggregator.best_practices = defaultdict(list, agg_state['best_practices'])
        
        self.meta_graph = checkpoint['meta_graph']
        
        self.log(f"Checkpoint loaded from {checkpoint_path}")
        self.log(f"Loaded {self.paper_count} papers")


# =============================================================================
# DOMAIN CLASSIFIER - Simple domain detection
# =============================================================================

class DomainClassifier:
    """Simple rule-based domain classifier."""
    
    def __init__(self):
        self.domain_keywords = {
            'NLP': ['transformer', 'bert', 'gpt', 'language', 'text', 'tokenizer', 'nlp'],
            'Vision': ['cnn', 'resnet', 'image', 'convolution', 'vision', 'detection'],
            'RL': ['reinforcement', 'policy', 'reward', 'agent', 'environment', 'dqn']
        }
    
    def classify(self, text: str) -> str:
        """Classify domain based on keywords."""
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        
        if max(scores.values()) == 0:
            return 'General'
        
        return max(scores, key=scores.get)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Knowledge Graph for Paper Reproduction'
    )
    parser.add_argument('--papers', type=str, nargs='+',
                        help='List of paper names (folder names)')
    parser.add_argument('--paper-base-path', type=str,
                        default='/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/short-papers',
                        help='Base path to papers directory')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for retriever')
    parser.add_argument('--log-path', type=str, default='continual_learning_output',
                        help='Path for log output')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load from checkpoint')
    parser.add_argument('--save-checkpoint', type=str, default=None,
                        help='Save checkpoint after processing')
    parser.add_argument('--max-code-files', type=int, default=-1,
                        help='Maximum code files to process per paper')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.log_path, exist_ok=True)
    log = Logger(args.log_path)
    
    base_url = "https://inference.airi.net:46783/v1"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3Njk0NDQ0MjQsImV4cCI6MTc3MDA0OTIyNH0.4eyt0_zsvfDY4HmwDsc4eS0p0mFDftFQL7u_DhRJqt4"
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
    
    log("="*70)
    log("CONTINUAL LEARNING KNOWLEDGE GRAPH")
    log("="*70)
    log(f"Model: {model}")
    log(f"Device: {args.device}")
    log(f"Papers to process: {len(args.papers) if args.papers else 0}")
    log("")
    
    # Initialize graph
    log("Initializing continual learning graph...")
    cl_graph = ContinualLearningGraph(
        model, 
        "You are a helpful assistant specializing in research reproduction",
        api_key, 
        log,
        base_url, 
        args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        log(f"Loading checkpoint from {args.checkpoint}...")
        cl_graph.load_checkpoint(args.checkpoint)
    
    # Process papers
    summaries = []
    
    for paper_name in args.papers:
        paper_path = os.path.join(args.paper_base_path, paper_name, 'paper.md')
        blacklist_path = os.path.join(args.paper_base_path, paper_name, 'blacklist.txt')
        
        # Load repo URL
        repo_url = None
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r') as f:
                repo_url = f.read().strip()
        
        if not repo_url or not repo_url.startswith('http'):
            log(f"Skipping {paper_name}: No valid repository URL")
            continue
        
        try:
            summary = cl_graph.process_paper(
                paper_path, 
                repo_url,
                max_code_files=args.max_code_files
            )
            summaries.append(summary)
        except Exception as e:
            log(f"Error processing {paper_name}: {str(e)}")
            import traceback
            log(traceback.format_exc())
            continue
    
    # Final analysis
    log("\n" + "="*70)
    log("CONTINUAL LEARNING SUMMARY")
    log("="*70)
    log(f"Total papers processed: {cl_graph.paper_count}")
    log(f"Total triplets: {len(cl_graph.triplets)}")
    log(f"Total unique entities: {len(cl_graph.aggregator.entity_frequencies)}")
    log(f"Total cost: ${cl_graph.total_amount:.4f}")
    log("")
    
    log("Top 20 most frequent entities:")
    for entity, count in cl_graph.aggregator.entity_frequencies.most_common(20):
        papers = len(cl_graph.entity_resolver.entity_metadata.get(entity, {}).get('papers', []))
        log(f"  {entity}: {count} mentions across {papers} papers")
    log("")
    
    log("Top 15 most frequent relations:")
    for relation, count in cl_graph.aggregator.relation_frequencies.most_common(15):
        log(f"  {relation}: {count} occurrences")
    log("")
    
    # Save checkpoint
    if args.save_checkpoint:
        cl_graph.save_checkpoint(args.save_checkpoint)
    
    # Save summary JSON
    summary_data = {
        'total_papers': cl_graph.paper_count,
        'papers_processed': cl_graph.papers_processed,
        'summaries': summaries,
        'total_triplets': len(cl_graph.triplets),
        'total_entities': len(cl_graph.aggregator.entity_frequencies),
        'total_cost': cl_graph.total_amount,
        'patterns': cl_graph.meta_graph.get('patterns', {}),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.log_path, 'continual_learning_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    log(f"\nSummary saved to {summary_path}")
    log("\nCONTINUAL LEARNING COMPLETE")


if __name__ == "__main__":
    main()
