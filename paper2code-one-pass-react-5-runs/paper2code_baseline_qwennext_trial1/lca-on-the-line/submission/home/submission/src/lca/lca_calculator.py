import numpy as np
import torch
import os
import json
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LCA_Calculator:
    """
    Calculates Lowest Common Ancestor (LCA) distance between predicted and true classes
    based on a predefined taxonomy hierarchy (e.g., WordNet).
    """
    
    def __init__(self, taxonomy_file: str):
        """
        Initialize with taxonomy hierarchy file (e.g., WordNet hierarchy)
        taxonomy_file: Path to JSON file containing class hierarchy
        """
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        self.class_to_node = {}
        self.node_to_class = {}
        self.node_to_depth = {}
        self._build_class_mappings()
    
    def _load_taxonomy(self, taxonomy_file: str) -> Dict:
        """Load taxonomy hierarchy from JSON file"""
        logger.info(f"Loading taxonomy from {taxonomy_file}")
        with open(taxonomy_file, 'r') as f:
            taxonomy = json.load(f)
        return taxonomy
    
    def _build_class_mappings(self):
        """Build mappings between class names and hierarchy nodes"""
        logger.info("Building class mappings from taxonomy")
        
        def traverse(node, path):
            if 'class' in node:
                class_name = node['class']
                self.class_to_node[class_name] = path
                self.node_to_class[path] = class_name
            if 'children' in node:
                for child in node['children']:
                traverse(child, path + (node['class'] if 'class' in node else 'root'))
        
        # Build mappings from taxonomy structure
        for root in self.taxonomy:
            traverse(root, ())
        
        logger.info(f"Built mappings for {len(self.class_to_node)} classes")
    
    def get_lca_distance(self, true_class: str, pred_class: str) -> float:
        """
        Calculate LCA distance between true class and predicted class
        Returns the taxonomic distance (path length from true to pred via LCA)
        """
        if true_class == pred_class:
            return 0.0
        
        if true_class not in self.class_to_node:
            logger.warning(f"True class '{true_class}' not found in taxonomy")
            return 10.0  # Default high distance
        if pred_class not in self.class_to_node:
            logger.warning(f"Pred class '{pred_class}' not found in taxonomy")
            return 10.0  # Default high distance
        
        true_path = self.class_to_node[true_class]
        pred_path = self.class_to_node[pred_class]
        
        # Find LCA by finding common prefix
        lca_path = []
        for i in range(min(len(true_path), len(pred_path))):
            if true_path[i] == pred_path[i]:
                lca_path.append(true_path[i])
            else:
                break
        
        # Calculate distance = depth(true) + depth(pred) - 2 * depth(LCA)
        true_depth = len(true_path)
        pred_depth = len(pred_path)
        lca_depth = len(lca_path)
        
        distance = float(true_depth + pred_depth - 2 * lca_depth)
        return distance
    
    def calculate_lca_for_batch(self, true_classes: List[str], pred_classes: List[str]) -> float:
        """
        Calculate average LCA distance for a batch
        """
        distances = []
        for true, pred in zip(true_classes, pred_classes):
            dist = self.get_lca_distance(true, pred)
            distances.append(dist)
        return float(np.mean(distances) if distances else 0.0)