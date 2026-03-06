#!/usr/bin/env python3
"""
Test script for building a reproduction-focused knowledge graph from academic papers.
Extracts implementation details to create a "cookbook" for reproducing research.
"""

import re
import os
import sys
import ast
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from time import time, sleep
from collections import Counter
import datetime

from graphs.contriever_graph import ContrieverGraph
from utils.utils import Logger


from pyvis.network import Network
import networkx as nx


sys.path.insert(0, os.path.dirname(__file__))
from prompts.paper_reproduction_prompt import prompt_extraction_reproduction, prompt_extraction_reproduction_with_repo
from prompts.cookbook_extraction_prompt import (
    prompt_cookbook_extraction,
    prompt_cookbook_extraction_wo_code,
    prompt_paper_code_linking,
    prompt_cookbook_reusable_extraction,
    prompt_reusable_pattern_to_code_mapping,
    prompt_coding_agent_with_cookbook,
    prompt_pattern_compatibility_check,
    BENCHMARK_EVALUATION_RULES,
    REPRODUCTION_SCRIPT_TOY_EXAMPLE,
)
from graph_storage import (
    load_cookbook_graph,
    save_cookbook_graph,
    get_cookbook_path,
    merge_triplets_into_cookbook,
    generate_hypernode_id,
)
from ontology.cookbook_ontology import validate_triplet


# =============================================================================
# REPOSITORY PROCESSING
# =============================================================================

# Common code file extensions
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sh', '.bash', '.yaml', '.yml', '.toml', '.json'
}

# Directories to skip
SKIP_DIRS = {
    '.git', '.github', '__pycache__', '.pytest_cache', 'venv', 'env',
    '.venv', '.next', '.nuxt', 
    '.idea', '.vscode', '.cache',
    'docs', 'examples', 'notebooks', 'data', 'assets', 'figures', 'tests'
}


def clone_repo(repo_url, target_dir, log):
    """Clone GitHub repository to target directory."""
    # Ensure the destination directory exists (the command will create the repo subfolder)
    destination_parent_dir = os.path.dirname(target_dir)
    os.makedirs(destination_parent_dir, exist_ok=True)

    log(f"Cloning {repo_url}...")
    subprocess.run(['git', 'clone', repo_url, target_dir], 
                   check=True, capture_output=True, #cwd=destination_parent_dir
                   )
    log(f"Repository cloned to {target_dir}")


def should_process_file(file_path):
    """Check if file should be included."""
    return file_path.suffix.lower() in CODE_EXTENSIONS


def collect_code_files(repo_dir, max_files=-1):
    """Walk through repository and collect code files."""
    code_files = []
    repo_path = Path(repo_dir)
    
    for file_path in repo_path.rglob('*'):
        if file_path.is_dir():
            continue
        if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRS):
            continue
        
        if should_process_file(file_path):
            try:
                rel_path = file_path.relative_to(repo_path)
                code_files.append((str(rel_path), file_path))
            except Exception:
                assert False
    
    # Sort by likely importance (main files first, then by size)
    def file_importance(item):
        rel_path, file_path = item
        score = 0
        
        # Prioritize main/core files
        if 'main' in rel_path.lower() or 'core' in rel_path.lower():
            score += 1000
        if 'model' in rel_path.lower() or 'train' in rel_path.lower():
            score += 500
        if 'config' in rel_path.lower() or 'setup' in rel_path.lower():
            score += 300
        
        # Deprioritize deeply nested files
        score -= rel_path.count('/') * 10
        
        # Add file size (larger files tend to be more important)
        try:
            score += min(file_path.stat().st_size / 100, 100)
        except:
            pass
        
        return -score  # Negative for reverse sort
    
    code_files.sort(key=file_importance)
    
    # Limit number of files
    if max_files > 0:
        code_files = code_files[:max_files]
    return code_files


def read_file_safe(file_path):
    """Read file with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "[Binary file - content not displayed]"


def extract_code_chunks(file_content, file_path, max_chunk_size=3000):
    """
    Extract meaningful chunks from code file.
    Focus on: classes, functions, important comments, config sections.
    """
    chunks = []
    
    # For Python files, extract classes and functions
    if file_path.endswith('.py'):
        # Split by class or function definitions
        pattern = r'((?:^|\n)(?:class |def |# ===|# ---)[^\n]*(?:\n(?:    |\t)[^\n]*)*)'
        matches = re.finditer(pattern, file_content, re.MULTILINE)
        
        for match in matches:
            chunk = match.group(0).strip()
            if len(chunk) > 100:  # Skip very small chunks
                chunks.append(chunk)
    
    # For config files (yaml, json, toml), take the whole content if small enough
    elif file_path.endswith(('.yaml', '.yml', '.json', '.toml', '.cfg', '.ini')):
        if len(file_content) < max_chunk_size:
            chunks.append(file_content)
        else:
            # Split into sections
            sections = file_content.split('\n\n')
            for section in sections:
                if len(section) > 100:
                    chunks.append(section)
    
    # For other files, use simple chunking
    else:
        # Split by double newlines or comments
        sections = re.split(r'\n\n+|(?:\n *//.*\n)+|(?:\n *#.*\n)+', file_content)
        for section in sections:
            if len(section) > 100:
                chunks.append(section)
    
    # Ensure chunks are not too large
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split large chunks
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i+max_chunk_size])
    
    return final_chunks#[:10]  # Max 10 chunks per file


def build_code_index(repo_dir, code_files, log=None):
    """
    Parse codebase to extract structured index: file path, type (class/function/config_key), name.
    Used for Phase 2 linking pass.
    """
    index = []
    repo_path = Path(repo_dir)

    for rel_path, file_path in code_files:
        if not file_path.exists():
            continue
        content = read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 10:
            continue

        rel_path_str = str(rel_path)

        # Python: use ast to extract class and function names
        if str(rel_path).endswith('.py'):
            try:
                tree = ast.parse(content)

                class _Visitor(ast.NodeVisitor):
                    def __init__(self):
                        self.entries = []
                        self._class_stack = []
                        self._in_function = False

                    def visit_ClassDef(self, node):
                        self.entries.append({
                            "file": rel_path_str,
                            "type": "class",
                            "name": node.name,
                            "qual": f"{rel_path_str}::{node.name}",
                        })
                        self._class_stack.append(node.name)
                        self.generic_visit(node)
                        self._class_stack.pop()

                    def visit_FunctionDef(self, node):
                        if self._in_function:
                            self.generic_visit(node)
                            return
                        self._in_function = True
                        if self._class_stack:
                            qual_name = f"{self._class_stack[-1]}.{node.name}"
                            ent_type = "method"
                        else:
                            qual_name = node.name
                            ent_type = "function"
                        self.entries.append({
                            "file": rel_path_str,
                            "type": ent_type,
                            "name": qual_name,
                            "qual": f"{rel_path_str}::{qual_name}",
                        })
                        self.generic_visit(node)
                        self._in_function = False

                v = _Visitor()
                v.visit(tree)
                index.extend(v.entries)
            except (SyntaxError, ValueError):
                for m in re.finditer(r'^\s*(?:class|def)\s+(\w+)', content, re.MULTILINE):
                    idx_type = "class" if m.group(0).strip().startswith("class") else "function"
                    index.append({
                        "file": rel_path_str,
                        "type": idx_type,
                        "name": m.group(1),
                        "qual": f"{rel_path_str}::{m.group(1)}",
                    })

        # Config files: extract top-level keys
        elif str(rel_path).endswith(('.json', '.yaml', '.yml')):
            try:
                if str(rel_path).endswith('.json'):
                    data = json.loads(content)
                else:
                    try:
                        import yaml
                        data = yaml.safe_load(content) or {}
                    except ImportError:
                        data = {}
                if isinstance(data, dict):
                    for key in list(data.keys())[:30]:
                        index.append({
                            "file": rel_path_str,
                            "type": "config",
                            "name": str(key),
                            "qual": f"{rel_path_str}::{key}",
                        })
            except (json.JSONDecodeError, Exception):
                pass
        elif str(rel_path).endswith('.toml'):
            try:
                try:
                    import tomllib
                    data = tomllib.loads(content)
                except ImportError:
                    try:
                        import tomli
                        data = tomli.loads(content)
                    except ImportError:
                        data = {}
                if isinstance(data, dict):
                    for key in data.keys():
                        index.append({
                            "file": rel_path_str,
                            "type": "config",
                            "name": key,
                            "qual": f"{rel_path_str}::{key}",
                        })
            except Exception:
                pass

    return index


# =============================================================================
# PAPER PROCESSING
# =============================================================================

def load_paper(paper_path):
    """Load the paper markdown file."""
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def load_repo_url(paper_dir):
    """Load repository URL from blacklist.txt in paper directory."""
    blacklist_path = os.path.join(paper_dir, 'blacklist.txt')
    if os.path.exists(blacklist_path):
        with open(blacklist_path, 'r') as f:
            url = f.read().strip()
            return url if url.startswith('http') else None
    return None


def split_into_sections(paper_content):
    """Split paper into sections based on \\section* delimiters."""
    section_pattern = r'\\section\*\{([^}]+)\}'
    sections = []
    
    matches = list(re.finditer(section_pattern, paper_content))
    
    for i, match in enumerate(matches):
        section_title = match.group(1)
        start_pos = match.end()
        
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(paper_content)
        
        section_content = paper_content[start_pos:end_pos].strip()
        
        if section_content:
            sections.append({
                'title': section_title,
                'content': section_content
            })
    
    if matches:
        preamble = paper_content[:matches[0].start()].strip()
        if preamble:
            sections.insert(0, {
                'title': 'Title and Abstract',
                'content': preamble
            })
    
    return sections


def clean_latex(text):
    """Clean LaTeX commands and formatting from text."""
    text = re.sub(r'\\title\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\author\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\begin\{abstract\}', '', text)
    text = re.sub(r'\\end\{abstract\}', '', text)
    text = re.sub(r'\\footnotetext\{[^}]+\}', '', text)
    text = re.sub(r'\\cite\{[^}]+\}', '', text)
    text = re.sub(r'\\ref\{[^}]+\}', '', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    
    # Keep equations but simplify
    text = re.sub(r'\$\$([^\$]+)\$\$', r'[\1]', text)
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    text = re.sub(r'\\\[([^\]]+)\\\]', r'[\1]', text)
    text = re.sub(r'\\\(([^\)]+)\\\)', r'\1', text)
    
    text = re.sub(r'!\[\]\([^)]+\)', '[Figure]', text)
    text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '[Table]', text, flags=re.DOTALL)
    
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{|\}', '', text)
    text = re.sub(r'\\\\', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def preprocess_section(section_content):
    """Preprocess a section by cleaning LaTeX and preparing for graph extraction."""
    cleaned = clean_latex(section_content)
    
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    
    max_chars = 2000  # Slightly longer chunks for reproduction details
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def analyze_reproduction_graph(graph):
    """Analyze the reproduction-focused knowledge graph."""
    triplets = graph.triplets
    
    stats = {
        'total_triplets': len(triplets),
        'unique_entities': set(),
        'entity_connections': Counter(),
        'relation_types': Counter(),
        'implementation_relations': Counter(),
        'configuration_relations': Counter(),
        'procedure_relations': Counter(),
    }
    
    # Categorize relations
    impl_keywords = ['implements', 'implemented', 'extends', 'uses', 'requires']
    config_keywords = ['parameter', 'configured', 'set to', 'default', 'value']
    proc_keywords = ['step', 'procedure', 'first', 'then', 'after']
    
    for triplet in triplets:
        subject, obj, rel_data = triplet
        relation = rel_data.get('label', '').lower()
        
        stats['unique_entities'].add(subject)
        stats['unique_entities'].add(obj)
        stats['entity_connections'][subject] += 1
        stats['entity_connections'][obj] += 1
        stats['relation_types'][rel_data.get('label', 'unknown')] += 1
        
        # Categorize relations
        if any(kw in relation for kw in impl_keywords):
            stats['implementation_relations'][relation] += 1
        if any(kw in relation for kw in config_keywords):
            stats['configuration_relations'][relation] += 1
        if any(kw in relation for kw in proc_keywords):
            stats['procedure_relations'][relation] += 1
    
    stats['num_unique_entities'] = len(stats['unique_entities'])
    stats['top_entities'] = stats['entity_connections'].most_common(15)
    stats['top_relations'] = stats['relation_types'].most_common(15)
    stats['top_impl_relations'] = stats['implementation_relations'].most_common(10)
    stats['top_config_relations'] = stats['configuration_relations'].most_common(10)
    
    return stats


class ReproductionGraph(ContrieverGraph):
    """Extended ContrieverGraph that uses reproduction-focused prompts."""

    def __init__(self, model, system_prompt, api_key, log, base_url='', device="cpu"):
        super().__init__(model, system_prompt, api_key, base_url, device)
        self.graph_builder_instruction = (
            "Extract only REUSABLE patterns for a cumulative cookbook. "
            "Skip paper-specific details. This graph helps implement OTHER papers in the area.\n\n"
        )
        self.reproduction_prompt = prompt_cookbook_reusable_extraction
        self.hypernode_store = {}
        self._log = log
        log(f'\nPrompt: {self.reproduction_prompt[:200]}...')

    def add_triplets(self, triplets):
        """Override to enforce ontology validation before adding."""
        validated = []
        for t in triplets:
            if validate_triplet(t):
                validated.append(t)
            else:
                self._log(f"Skipping invalid triplet (ontology): {t}")
        if validated:
            super().add_triplets(validated)
    
    def update_without_retrieve(self, observation, prev_subgraph, log, source_type="paper"):
        """Override to use reproduction-focused prompt."""
        example = [re.sub(r"Step \d+: ", "", t) for t in prev_subgraph]
        example_str = "; ".join(example) if example else ""
        
        if source_type == "code":
            observation = f"[CODE IMPLEMENTATION]\n{observation}"
        else:
            observation = f"[PAPER TEXT]\n{observation}"
        
        prompt = self.graph_builder_instruction + self.reproduction_prompt.format(
            observation=observation, 
            example=example_str
        )
        
        response, _ = self.generate(prompt, t=0.001)
        #log('response generated')
        # Process triplets (reuse parent class logic)
        from utils.utils import process_triplets
        new_triplets_raw = process_triplets(response)
        self.add_triplets(new_triplets_raw)
        #log('triplets processed')
        
        new_triplets = self.exclude(new_triplets_raw)
        n_added = len(new_triplets)
        n_parsed = len(new_triplets_raw)
        if n_added < n_parsed:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new (added to graph), {n_parsed - n_added} duplicates or invalid")
        else:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new")
        
        obs_embedding = self.retriever.embed(observation)
        
        new_triplets_str = self.convert(new_triplets_raw)
        obs_value = [new_triplets_str, obs_embedding]
        self.obs_episodic[observation] = obs_value
        return new_triplets_raw, obs_value


def vis_net(subgraph, exp_path, save_as=''):
    """
    Visualize graph with multiple connected components properly separated.
    Each component is positioned independently to prevent overlap.
    """
    import math
    # Find connected components within the subgraph
    components = list(nx.connected_components(subgraph))
    num_comps = len(components)
    # Create pyvis network
    net = Network(
        height="1000px",
        width="100%",
        bgcolor="#ffffff",
        font_color='#10000000' if num_comps < 2 else "black",
        directed=True,
    )

    
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(components) > len(colors):
        colors = list(mcolors.CSS4_COLORS.values())

    # Assign colors to nodes based on component
    component_map = {}
    for i, component in enumerate(components):
        color = colors[i % len(colors)]
        for node in component:
            component_map[node] = {'component_id': i, 'color': color}

    # Calculate layout for each component separately with proper spacing
    all_positions = {}
    component_spacing = 2000  # Space between components
    
    # Arrange components in a grid layout
    grid_cols = math.ceil(math.sqrt(len(components)))
    
    for idx, component in enumerate(components):
        # Create subgraph for this component
        comp_subgraph = subgraph.subgraph(component)
        
        # Choose layout algorithm based on component size
        if len(component) < 30:
            # For small components, use spring layout
            pos = nx.spring_layout(comp_subgraph, k=2, iterations=50, scale=500)
        elif len(component) < 100:
            # For medium components, use spring layout with more space
            pos = nx.spring_layout(comp_subgraph, k=1.5, iterations=50, scale=800)
        else:
            # For large components, use kamada_kawai or spring with large scale
            try:
                pos = nx.kamada_kawai_layout(comp_subgraph, scale=1000)
            except:
                pos = nx.spring_layout(comp_subgraph, k=1, iterations=50, scale=1000)
        
        # Calculate offset for this component in the grid
        row = idx // grid_cols
        col = idx % grid_cols
        offset_x = col * component_spacing
        offset_y = row * component_spacing
        
        # Apply offset to separate components
        for node, (x, y) in pos.items():
            all_positions[node] = (x + offset_x, y + offset_y)
    
    # Add nodes with fixed positions
    for node in subgraph.nodes():
        x, y = all_positions[node]
        net.add_node(
            node,
            label=node,
            title=node,
            color=component_map[node]['color'],
            shape='dot',
            size=10,
            x=x,
            y=y,
            physics=False,  # Fix position
        )

    # Add edges
    for s, o, edge_data in subgraph.edges(data=True):
        net.add_edge(
            s, o, 
            label=edge_data.get('label', ''),
            title=edge_data.get('label', ''),
        )
    
    # Disable physics since we have fixed positions
    net.toggle_physics(False)
    
    # Set options for better visualization
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": false,
          "color": "#848484"
        },
        "smooth": {
          "type": "continuous"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "font": {
          "size": 10,
          "align": "middle"
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Save and display
    filename = f'{exp_path}/{save_as}.html'
    net.save_graph(filename)


def _triplet_to_line(triplet):
    """Convert internal triplet representation to prompt-friendly text."""
    subj, obj, rel = triplet
    rel_label = rel.get('label', 'related_to') if isinstance(rel, dict) else str(rel)
    return f"{subj}, {rel_label}, {obj}"


def _extract_json_blocks(text):
    """Extract JSON objects from LLM response (handles multiple blocks)."""
    blocks = []
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    blocks.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return blocks


def run_linking_pass_hypernodes(paper_triplets, code_index, code_files, graph, log, max_paper_triplets=None):
    """
    Phase 2 linking (hypernode mode): for each code file, use LLM to produce hypernodes
    linking reusable patterns to code. Parses JSON blocks, stores in hypernode_store,
    adds (pattern, implemented_in, hypernode_id) triplets.
    """
    paper_lines = _get_context_triplets(paper_triplets, max_triplets=max_paper_triplets)
    reusable_patterns_block = "\n".join(f"- {line}" for line in paper_lines)

    hypernode_store = getattr(graph, "hypernode_store", {})
    if not isinstance(hypernode_store, dict):
        hypernode_store = {}
        graph.hypernode_store = hypernode_store

    hypernodes_added = 0

    for rel_path, file_path in code_files:
        if not file_path.exists():
            continue
        content = read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 10:
            continue

        rel_path_str = str(rel_path)
        file_index = [e for e in code_index if e["file"] == rel_path_str]
        if not file_index:
            file_index = [{"file": rel_path_str, "type": "file", "name": rel_path_str, "qual": rel_path_str}]

        index_lines = [f"- {e['qual']} ({e['type']})" for e in file_index]
        index_block = "\n".join(index_lines[:50])

        chunks = extract_code_chunks(content, rel_path_str)
        code_chunk = "\n\n---\n\n".join(chunks[:3]) if chunks else content[:2000]
        code_chunk = code_chunk[:4000]

        prompt = prompt_reusable_pattern_to_code_mapping.format(
            reusable_patterns=reusable_patterns_block,
            code_index=index_block,
            file_path=rel_path_str,
            code_chunk=code_chunk,
        )

        try:
            response, _ = graph.generate(prompt, t=0.001)
            blocks = _extract_json_blocks(response)
            for block in blocks:
                pattern = block.get("pattern")
                hypernode = block.get("hypernode")
                if not pattern or not hypernode:
                    continue
                code_val = hypernode.get("code", "")
                doc = hypernode.get("documentation", "")
                imports = hypernode.get("imports", [])
                if not isinstance(imports, list):
                    imports = [imports] if imports else []
                if not code_val and not doc:
                    continue
                hn = {"code": code_val, "documentation": doc, "imports": imports}
                hid = generate_hypernode_id(pattern, code_val)
                if hid not in hypernode_store:
                    hypernode_store[hid] = hn
                    graph.add_triplets([[pattern, hid, {"label": "implemented_in"}]])
                    hypernodes_added += 1
                    log(f"  Hypernode: {pattern} -> {hid} ({rel_path_str})")
        except Exception as e:
            log(f"  Error linking {rel_path_str}: {e}")

    return hypernodes_added


def run_linking_pass(paper_triplets, code_index, code_files, graph, log, max_paper_triplets=None):
    """
    Phase 2 linking (legacy path-based): for each code file, use LLM to create triplets
    linking paper entities to code paths. Uses prompt_paper_code_linking.
    """
    from utils.utils import process_triplets

    paper_lines = _get_context_triplets(paper_triplets, max_triplets=max_paper_triplets)
    paper_block = "\n".join(f"- {line}" for line in paper_lines)

    linking_triplets_total = 0

    for rel_path, file_path in code_files:
        if not file_path.exists():
            continue
        content = read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 10:
            continue

        rel_path_str = str(rel_path)
        file_index = [e for e in code_index if e["file"] == rel_path_str]
        if not file_index:
            file_index = [{"file": rel_path_str, "type": "file", "name": rel_path_str, "qual": rel_path_str}]

        index_lines = [f"- {e['qual']} ({e['type']})" for e in file_index]
        index_block = "\n".join(index_lines[:50])

        chunks = extract_code_chunks(content, rel_path_str)
        code_chunk = "\n\n---\n\n".join(chunks[:3]) if chunks else content[:2000]
        code_chunk = code_chunk[:4000]

        prompt = prompt_paper_code_linking.format(
            paper_graph_triplets=paper_block,
            code_index=index_block,
            file_path=rel_path_str,
            code_chunk=code_chunk,
        )

        try:
            response, _ = graph.generate(prompt, t=0.001)
            new_triplets_raw = process_triplets(response)
            graph.add_triplets(new_triplets_raw)
            linking_triplets_total += len(new_triplets_raw)
            if new_triplets_raw:
                log(f"  Linking: {len(new_triplets_raw)} triplets for {rel_path_str}")
                for t in new_triplets_raw[:2]:
                    log(f"    - {_triplet_to_line(t)}")
        except Exception as e:
            log(f"  Error linking {rel_path_str}: {e}")

    return linking_triplets_total


def _get_context_triplets(triplets, max_triplets=None, recent_first=False):
    """
    Get triplets formatted for prompt context.
    If max_triplets is None, include all triplets. Otherwise limit count.
    If recent_first=True and max_triplets is set, take the most recent triplets (from end of list).
    Deduplicates by line.
    """
    if not triplets:
        return []
    lines = []
    seen = set()
    it = reversed(triplets) if (recent_first and max_triplets is not None) else triplets
    for t in it:
        line = _triplet_to_line(t)
        if line not in seen and (max_triplets is None or len(lines) < max_triplets):
            seen.add(line)
            lines.append(line)
    if recent_first and max_triplets is not None:
        lines = list(reversed(lines))  # Restore chronological order for prompt
    return lines


def _extract_python_code(text):
    """Extract Python code from fenced output, fallback to raw text."""
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_code_generator_output(response_text, repo_root):
    """
    Parse LLM output in format:
      FILE: path/to/file.ext
      ```language
      <content>
      ```

    Creates the repo structure under repo_root, preserves file names, extensions, and content.
    Returns (files_created: list of Path, primary_python: Path or None).
    """
    files_created = []
    primary_python = None

    # Pattern: FILE: path (allow spaces, forward/back slashes)
    file_decl_re = re.compile(r"^FILE:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
    # Fenced code block: ```lang\n...``` (content can span lines)
    code_block_re = re.compile(r"^```(\w*)\s*\n([\s\S]*?)```", re.MULTILINE)

    # Find all FILE: declarations with their positions
    file_matches = list(file_decl_re.finditer(response_text))

    for i, file_match in enumerate(file_matches):
        raw_path = file_match.group(1).strip()
        # Normalize: forward slashes, strip leading slashes
        raw_path = raw_path.replace("\\", "/").lstrip("/")
        if not raw_path:
            continue
        # Sanitize: avoid path traversal
        parts = raw_path.split("/")
        safe_parts = [p for p in parts if p and p != ".."]
        rel_path = "/".join(safe_parts) if safe_parts else raw_path
        if not rel_path:
            continue

        # Find the next code block after this FILE: line
        start = file_match.end()
        next_file_start = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(response_text)
        segment = response_text[start:next_file_start]
        code_match = code_block_re.search(segment)
        if not code_match:
            continue
        content = code_match.group(2).rstrip()
        if content.endswith("\n"):
            content = content[:-1]  # Keep trailing newline behavior consistent

        out_path = Path(repo_root) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(content, encoding="utf-8")
        except Exception:
            out_path.write_text(content, encoding="utf-8", errors="replace")
        files_created.append(out_path)
        if primary_python is None and out_path.suffix.lower() in (".py", ".pyw"):
            primary_python = out_path

    # Fallback: if no FILE: blocks found, try single fenced code block (legacy)
    if not files_created:
        for match in code_block_re.finditer(response_text):
            content = match.group(2).rstrip()
            lang = (match.group(1) or "").lower()
            ext = ".py" if "python" in lang or lang == "py" else ".sh" if "bash" in lang or "sh" in lang else ".txt"
            out_path = Path(repo_root) / f"generated_reproduction{ext}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out_path.write_text(content, encoding="utf-8")
            except Exception:
                out_path.write_text(content, encoding="utf-8", errors="replace")
            files_created.append(out_path)
            primary_python = out_path if ext == ".py" else primary_python
            break
        if not files_created:
            # Last resort: raw text as single file
            out_path = Path(repo_root) / "generated_reproduction.py"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(response_text.strip(), encoding="utf-8", errors="replace")
            files_created.append(out_path)
            primary_python = out_path

    return files_created, primary_python


def _format_triplets_for_prompt(triplets, max_triplets=None):
    """Format triplets as bullet list for LLM prompt."""
    lines = []
    for row in triplets[:max_triplets if max_triplets is not None else len(triplets)]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            s, o, r = row[0], row[1], row[2]
            rel = r.get("label", r) if isinstance(r, dict) else str(r)
            lines.append(f"- {s}, {rel}, {o}")
        elif isinstance(row, tuple) and len(row) == 3:
            s, o, r = row
            rel = r.get("label", r) if isinstance(r, dict) else str(r)
            lines.append(f"- {s}, {rel}, {o}")
    return "\n".join(lines) if lines else "(none)"


def _format_cookbook_for_agent(graph, hypernode_store, max_triplets=1500):
    """
    Format graph triplets + hypernode content for the coding agent.
    For each (pattern, implemented_in, hypernode_id), expand with code, docs, imports.
    """
    parts = []
    # Pattern/config triplets (exclude implemented_in for the patterns section)
    pattern_lines = []
    impl_triplets = []  # (pattern, hypernode_id)
    for triplet in graph.triplets:
        subj, obj, rel_data = triplet
        rel = rel_data.get("label", "") if isinstance(rel_data, dict) else str(rel_data)
        line = _triplet_to_line(triplet)
        if rel == "implemented_in":
            impl_triplets.append((subj, obj))
        else:
            pattern_lines.append(line)

    seen_patterns = set()
    unique_pattern_lines = []
    for line in pattern_lines:
        if line not in seen_patterns:
            seen_patterns.add(line)
            unique_pattern_lines.append(line)

    parts.append("## Patterns and config")
    parts.append("\n".join(f"- {line}" for line in unique_pattern_lines[:max_triplets]))

    # Expand hypernodes
    if hypernode_store and impl_triplets:
        parts.append("\n## Pattern -> Code (hypernodes)")
        for pattern, hid in impl_triplets:
            if hid not in hypernode_store:
                continue
            hn = hypernode_store[hid]
            code = hn.get("code", "")
            doc = hn.get("documentation", "")
            imports = hn.get("imports", [])
            if not isinstance(imports, list):
                imports = [imports] if imports else []
            parts.append(f"\n### {pattern} (implemented_in)")
            parts.append(f"Documentation: {doc}")
            parts.append(f"Imports: {', '.join(imports)}")
            if code:
                parts.append(f"Code:\n```\n{code[:3000]}\n```")  # Cap code length

    return "\n".join(parts)


def _format_expert_graph_for_agent(triplets, hypernode_store, max_triplets=None):
    """Format expert triplets + hypernodes for the coding agent (same logic as _format_cookbook_for_agent)."""
    parts = []
    pattern_lines = []
    impl_triplets = []
    for row in triplets[:max_triplets if max_triplets is not None else len(triplets)]:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            s, o, r = row[0], row[1], row[2]
        else:
            continue
        rel = r.get("label", r) if isinstance(r, dict) else str(r)
        line = f"{s}, {rel}, {o}"
        if rel == "implemented_in":
            impl_triplets.append((s, o))
        else:
            pattern_lines.append(line)

    seen = set()
    unique = []
    for line in pattern_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    parts.append("## Patterns and config")
    parts.append("\n".join(f"- {line}" for line in unique[:max_triplets if max_triplets is not None else len(unique)]))

    if hypernode_store and impl_triplets:
        parts.append("\n## Pattern -> Code (hypernodes)")
        for pattern, hid in impl_triplets:
            if hid not in hypernode_store:
                continue
            hn = hypernode_store[hid]
            code = hn.get("code", "")
            doc = hn.get("documentation", "")
            imports = hn.get("imports", [])
            if not isinstance(imports, list):
                imports = [imports] if imports else []
            parts.append(f"\n### {pattern} (implemented_in)")
            parts.append(f"Documentation: {doc}")
            parts.append(f"Imports: {', '.join(imports)}")
            if code:
                parts.append(f"Code:\n```\n{code}\n```")
    return "\n".join(parts)


def generate_code_from_graph(graph, paper_path, log, output_dir, max_triplets=None,
                             paper_summary=None, hypernode_store=None):
    """
    Generate implementation code using paper graph + expert graph.
    Loads paper_graph_data.json (paper details) and graph_data.json (expert knowledge) from output_dir.
    Falls back to graph + paper_path if files are missing.
    """
    paper_graph_path = os.path.join(output_dir, "paper_graph_data.json")
    graph_data_path = os.path.join(output_dir, "graph_data.json")

    paper_graph_block = ""
    expert_graph_block = ""

    if os.path.exists(paper_graph_path) and os.path.exists(graph_data_path):
        try:
            with open(paper_graph_path, "r", encoding="utf-8") as f:
                paper_data = json.load(f)
            with open(graph_data_path, "r", encoding="utf-8") as f:
                expert_data = json.load(f)
            paper_triplets = paper_data.get("triplets", [])
            expert_triplets = expert_data.get("triplets", [])
            expert_hypernodes = expert_data.get("hypernode_store", {})
            paper_graph_block = _format_triplets_for_prompt(paper_triplets, max_triplets)
            expert_graph_block = _format_expert_graph_for_agent(
                expert_triplets, expert_hypernodes, max_triplets
            )
            log(f"Loaded paper graph ({len(paper_triplets)} triplets) and expert graph ({len(expert_triplets)} triplets) for code generation")
        except (json.JSONDecodeError, KeyError) as e:
            log(f"Could not load graphs: {e}, falling back to in-memory graph")
            paper_graph_block = ""
            expert_graph_block = ""
    '''
    if not paper_graph_block or not expert_graph_block:
        # Fallback: use in-memory graph (no paper_graph.json / graph_data.json)
        if graph:
            paper_triplets = [[t[0], t[1], t[2]] for t in graph.triplets]
            hs = hypernode_store if hypernode_store is not None else getattr(graph, "hypernode_store", {})
            paper_graph_block = _format_triplets_for_prompt(paper_triplets, max_triplets)
            expert_graph_block = _format_cookbook_for_agent(graph, hs, max_triplets)
            log("Using in-memory graph (paper_graph.json / graph_data.json not found)")
    '''
    if not paper_summary and paper_path:
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                full = f.read()
            paper_summary = full[:3000] + ("..." if len(full) > 3000 else "")
        except Exception:
            pass
    if not paper_summary:
        paper_summary = f"Paper: {os.path.basename(paper_path)} (content not loaded)"

    prompt = prompt_coding_agent_with_cookbook.format(
        BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
        TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        paper_summary=paper_summary,
        paper_graph=paper_graph_block,
        expert_graph=expert_graph_block,
    )

    response, _ = graph.generate(prompt, t=0.2)

    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    repo_root = os.path.join(output_dir, "submission")
    os.makedirs(repo_root, exist_ok=True)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    files_created, primary_python = _parse_code_generator_output(response, repo_root)
    code_path = str(primary_python) if primary_python else (os.path.join(repo_root, "generated_reproduction.py") if files_created else None)

    for p in files_created:
        try:
            rel = Path(p).relative_to(Path(repo_root))
        except ValueError:
            rel = p
        log(f"  Created: {rel}")
    log(f"Generated repo at {repo_root} ({len(files_created)} files)")
    log(f"Raw model response saved to: {raw_output_path}")

    return {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created],
        "raw_response_path": raw_output_path,
        "triplets_used": len(paper_graph_block) + len(expert_graph_block),
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()),
    }

'''
def test_generated_code_executability(code_path, log, timeout_sec=20):
    """Run lightweight executability checks for generated Python code."""
    results = {
        "compile_check": {"success": False, "stdout": "", "stderr": ""},
        "smoke_run": {"success": False, "stdout": "", "stderr": ""},
    }

    try:
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", code_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        results["compile_check"] = {
            "success": compile_result.returncode == 0,
            "stdout": compile_result.stdout[-4000:],
            "stderr": compile_result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as e:
        results["compile_check"] = {
            "success": False,
            "stdout": (e.stdout or "")[-4000:],
            "stderr": f"Timeout after {timeout_sec}s",
        }

    if results["compile_check"]["success"]:
        try:
            smoke_result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=os.path.dirname(code_path),
            )
            results["smoke_run"] = {
                "success": smoke_result.returncode == 0,
                "stdout": smoke_result.stdout[-4000:],
                "stderr": smoke_result.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as e:
            results["smoke_run"] = {
                "success": False,
                "stdout": (e.stdout or "")[-4000:],
                "stderr": f"Timeout after {timeout_sec}s",
            }

    log("Generated code executability checks:")
    log(f"  - Compile check: {'PASS' if results['compile_check']['success'] else 'FAIL'}")
    log(f"  - Smoke run: {'PASS' if results['smoke_run']['success'] else 'FAIL'}")
    if results["compile_check"]["stderr"]:
        log(f"  - Compile stderr: {results['compile_check']['stderr'][:500]}")
    if results["smoke_run"]["stderr"]:
        log(f"  - Smoke stderr: {results['smoke_run']['stderr'][:500]}")

    return results
'''

def run_reproduction_test(paper_path, device="cpu", log_path="",
max_code_files=-1, generate_code=True, test_generated_code=True, repo_url_override=None, use_repo=True,
cookbook_path=None, load_cookbook=True, repo_dir=None):
    """Main function to run reproduction-focused paper test."""
    CONTEXT_WINDOW_LENGTHS: dict[str, int] = {
    "Qwen/QwQ-32B": 16384,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 262144,
    }
    
    
    
    log = Logger(log_path)
    
    base_url = "https://inference.airi.net:46783/v1"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzIwNjUyNjAsImV4cCI6MTc3MjY3MDA2MH0.AhqeW-tV9jFHrTEFDHauMWIDBVop0Ht8B5UW_y7-fDg"
    model = 'Qwen/Qwen3-Next-80B-A3B-Instruct'#'Qwen/QwQ-32B'#'Qwen/Qwen3-Coder-30B-A3B-Instruct'

    # Get paper directory and load repo URL
    paper_dir = os.path.dirname(paper_path)
    if not use_repo:
        repo_url = None
    elif repo_url_override:
        repo_url = repo_url_override
    else:
        repo_url = load_repo_url(paper_dir)
    
    cookbook_file = cookbook_path or get_cookbook_path()
    existing_triplets = []
    hypernode_store = {}
    existing_metadata = {}

    if load_cookbook and os.path.exists(cookbook_file):
        log(f"Loading cookbook from {cookbook_file}...")
        existing_triplets, hypernode_store, existing_metadata = load_cookbook_graph(cookbook_file)
        log(f"Loaded {len(existing_triplets)} triplets, {len(hypernode_store)} hypernodes")
    else:
        log("Starting with empty cookbook (--no-load-cookbook or file not found)")

        log("="*70)
        log("REPRODUCTION COOKBOOK KNOWLEDGE GRAPH BUILDER")
        log("="*70)
    log(f"Paper: {paper_path}")
    log(f"Repository: {repo_url or 'Not specified'}")
    log(f"Model: {model}")
    log(f"Device: {device}")
    log(f"Cookbook: {cookbook_file}")
    log("")

    total_start_time = time()

    log("Initializing reproduction-focused knowledge graph...")
    graph = ReproductionGraph(model, "You are a helpful assistant specializing in research reproduction",
                             api_key, log, base_url, device)
    # Pre-populate with existing cookbook
    for t in existing_triplets:
        graph.add_triplets([t])
    graph.hypernode_store = hypernode_store
    log("Graph initialized with reproduction-focused prompt")
    log("")

    # Check if graph was already built for this paper (skip regeneration)
    graph_data_path = os.path.join(log_path, "graph_data.json")
    paper_graph_path = os.path.join(log_path, "paper_graph_data.json")
    if os.path.exists(graph_data_path):
        try:
            with open(graph_data_path, "r") as f:
                cached = json.load(f)
            if os.path.exists(paper_graph_path):
                with open(paper_graph_path, "r") as f:
                    paper_cached = json.load(f)
                expert_triplets = cached.get("triplets", [])
                paper_triplets = paper_cached.get("triplets", [])
                graph.triplets = []
                for row in expert_triplets + paper_triplets:
                    s, o, r = row[0], row[1], row[2]
                    if isinstance(r, str):
                        r = {"label": r}
                    graph.add_triplets([(s, o, r)])
                graph.hypernode_store = cached.get("hypernode_store", hypernode_store)
                section_stats = paper_cached.get("sections", [])
                code_stats = paper_cached.get("code_stats", [])
                total_time = paper_cached.get("stats", {}).get("total_time", 0)
                log(f"Loaded cached: paper graph ({len(paper_triplets)} triplets) + expert graph ({len(expert_triplets)} triplets)")
            else:
                assert False, "Paper graph data file not found"
                '''
                # Legacy format: full graph in graph_data.json
                loaded_triplets = cached.get("triplets", [])
                graph.triplets = []
                for row in loaded_triplets:
                    s, o, r = row[0], row[1], row[2]
                    if isinstance(r, str):
                        r = {"label": r}
                    graph.add_triplets([(s, o, r)])
                graph.hypernode_store = cached.get("hypernode_store", hypernode_store)
                section_stats = cached.get("sections", [])
                code_stats = cached.get("code_stats", [])
                total_time = cached.get("stats", {}).get("total_time", 0)
                log(f"Loaded cached graph from {graph_data_path} ({len(loaded_triplets)} triplets)")
                '''
            log("Skipping Phase 1 (paper extraction) and Phase 2 (code linking)")
            skip_regeneration = True
        except (json.JSONDecodeError, KeyError) as e:
            log(f"Could not load cached graph: {e}, regenerating")
            skip_regeneration = False
    else:
        skip_regeneration = False

    if not skip_regeneration:
        log("Loading paper...")
        paper_content = load_paper(paper_path)
        sections = split_into_sections(paper_content)
        log(f"Paper split into {len(sections)} sections")
        log("")

        # Process sections
        section_stats = []

        for i, section in enumerate(sections):
            section_start_time = time()
            current_datetime = datetime.datetime.now()
            log("="*70)
            log(f"{current_datetime} SECTION {i+1}/{len(sections)}: {section['title']}")
            log("="*70)

            chunks = preprocess_section(section['content'])
            log(f"Section split into {len(chunks)} chunks")

            section_triplets = []

            for j, chunk in enumerate(chunks):
                log(f"\n{datetime.datetime.now()} Processing chunk {j+1}/{len(chunks)}...")
                log(f"Chunk preview: {chunk[:100]}...")

                # Pass recent triplets as context for entity consistency across chunks
                prev_subgraph = _get_context_triplets(graph.triplets, max_triplets=40, recent_first=True)

                retries = 3
                while retries > 0:
                    try:
                        new_triplets, _ = graph.update_without_retrieve(chunk, prev_subgraph, log,
                        source_type="paper")
                        section_triplets.extend(new_triplets)
                        log(f"{datetime.datetime.now()} Chunk {j+1}/{len(chunks)} done")
                        if new_triplets:
                            log("Sample triplets:")
                            for triplet in new_triplets[:3]:
                                subj, obj, rel = triplet
                                log(f"  - ({subj}) --[{rel.get('label', 'N/A')}]--> ({obj})")
                        break
                    except Exception as e:
                        log(f"Error processing chunk: {str(e)}")
                        retries -= 1
                        if retries > 0:
                            log(f"Retrying ({retries} left)...")
                            sleep(5)
                        else:
                            raise e

            section_time = time() - section_start_time

            log("")
            log(f"Section summary:")
            log(f"  - Triplets extracted: {len(section_triplets)}")
            log(f"  - Total triplets in graph: {len(graph.triplets)}")
            log(f"  - Processing time: {section_time:.2f} seconds")
            log(f"  - API cost so far: ${graph.total_amount:.4f}")
            log("")

            # Save current graph state after each section (allows stopping and inspecting)
            log.to_json({
                "paper": os.path.basename(paper_path),
                "sections_processed": i + 1,
                "triplets": [[t[0], t[1], t[2]] for t in graph.triplets],
                "hypernode_store": getattr(graph, "hypernode_store", {}),
                "stats": {"total_triplets": len(graph.triplets), "api_cost": graph.total_amount},
            }, "graph_data.json")
            log("Graph state saved to graph_data.json")

            section_stats.append({
                'section_num': i + 1,
                'title': section['title'],
                'triplets_extracted': len(section_triplets),
                'processing_time': section_time,
                'triplets': [[t[0], t[1], t[2]] for t in section_triplets],
            })

        # =============================================================================
        # PHASE 2: Code Index + Paper-to-Code Linking
        # =============================================================================
        paper_graph_triplets = list(graph.triplets)  # Snapshot after Phase 1

        code_stats = []
        if repo_dir is not None or repo_url is not None:
            log("="*70)
            log("PHASE 2: CODE INDEX + PAPER-TO-CODE LINKING")
            log("="*70)

            if repo_dir is not None:
                temp_dir = repo_dir
                log(f'Using local repo at {temp_dir}')
            else:
                temp_dir = log_path + '/repo'
                try:
                    if not os.path.exists(temp_dir):
                        log('Cloning repo...')
                        clone_repo(repo_url, temp_dir, log)
                    else:
                        log('Repo already downloaded, processing...')
                except subprocess.CalledProcessError as e:
                    log(f"Error cloning repository: {e}")
                    raise
            try:

                log("\nCollecting code files...")
                code_files = collect_code_files(temp_dir, max_files=max_code_files)
                log(f"Found {len(code_files)} code files")

                log("\nBuilding code index...")
                code_index = build_code_index(temp_dir, code_files, log)
                log(f"Code index: {len(code_index)} entities (classes, functions, config keys)")

                log("\nRunning linking pass (hypernodes: patterns -> code)...")
                link_start = time()
                hypernodes_added = run_linking_pass_hypernodes(
                    paper_graph_triplets, code_index, code_files, graph, log, max_paper_triplets=None
                )
                link_time = time() - link_start
                log(f"Linking complete: {hypernodes_added} hypernodes in {link_time:.2f}s")
                log(f"Total cost so far: ${graph.total_amount:.4f}")

                code_stats = [{
                    'phase': 'linking',
                    'hypernodes_added': hypernodes_added,
                    'processing_time': link_time,
                    'code_index_size': len(code_index),
                }]

            except Exception as e:
                log(f"Error in Phase 2: {e}")
                raise
        else:
            log('No repo provided, skipping Phase 2 (code linking).')

        # Save paper graph (paper-specific triplets only) for code generation
        initial_count = len(existing_triplets)
        paper_only_triplets = graph.triplets[initial_count:] if initial_count > 0 else list(graph.triplets)
        log.to_json({
            "paper": os.path.basename(paper_path),
            "triplets": [[t[0], t[1], t[2]] for t in paper_only_triplets],
            "sections": section_stats,
            "code_stats": code_stats,
        }, "paper_graph.json")
        log("Paper graph saved to paper_graph.json")

        total_time = time() - total_start_time
    
        # Final analysis
        log("="*70)
        log("REPRODUCTION COOKBOOK ANALYSIS")
        log("="*70)
        
        stats = analyze_reproduction_graph(graph)
        
        log(f"Total processing time: {total_time:.2f} seconds")
        log(f"Total API cost: ${graph.total_amount:.4f}")
        log(f"Total triplets extracted: {stats['total_triplets']}")
        log(f"Unique components/concepts: {stats['num_unique_entities']}")
        log("")
        
        log("Top 15 most connected components (key implementation elements):")
        for entity, count in stats['top_entities']:
            log(f"  - {entity}: {count} connections")
        log("")
    
        log("Top 15 relation types:")
        for relation, count in stats['top_relations']:
            log(f"  - {relation}: {count} occurrences")
        log("")
        
        if stats['top_impl_relations']:
            log("Top implementation relations:")
            for relation, count in stats['top_impl_relations']:
                log(f"  - {relation}: {count} occurrences")
            log("")
        
        if stats['top_config_relations']:
            log("Top configuration relations:")
            for relation, count in stats['top_config_relations']:
                log(f"  - {relation}: {count} occurrences")
            log("")
        '''
        log("Section-by-section breakdown:")
        for stat in section_stats:
            log(f"  Section {stat['section_num']}: {stat['title']}")
            log(f"    Triplets: {stat['triplets_extracted']}, Time: {stat['processing_time']:.2f}s")
        log("")
        '''
        graph_data = {
            'paper': os.path.basename(paper_path),
            'repository': repo_url,
            'triplets': [[t[0], t[1], t[2]] for t in graph.triplets],
            'hypernode_store': getattr(graph, "hypernode_store", {}),
            'stats': {
                'total_triplets': stats['total_triplets'],
                'num_unique_entities': stats['num_unique_entities'],
                'total_time': total_time,
                'total_cost': graph.total_amount,
            },
            'sections': section_stats,
            'code_stats': code_stats,
        }
    else:
        # Skip path: build graph_data from loaded graph for visualization
        stats = analyze_reproduction_graph(graph)
        graph_data = {
            'paper': os.path.basename(paper_path),
            'repository': repo_url,
            'triplets': [[t[0], t[1], t[2]] for t in graph.triplets],
            'hypernode_store': getattr(graph, "hypernode_store", {}),
            'stats': {
                'total_triplets': stats['total_triplets'],
                'num_unique_entities': stats['num_unique_entities'],
                'total_time': total_time,
                'total_cost': graph.total_amount,
            },
            'sections': section_stats,
            'code_stats': code_stats,
        }

    # Merge and persist cookbook (skip when loaded from cache)
    if not skip_regeneration:
        initial_count = len(existing_triplets)
        new_triplets = graph.triplets[initial_count:] if initial_count > 0 else graph.triplets
        new_hypernodes = {k: v for k, v in getattr(graph, "hypernode_store", {}).items()
                          if k not in hypernode_store}
        if new_triplets or new_hypernodes:
            merged_triplets, merged_hypernodes = merge_triplets_into_cookbook(
                existing_triplets, new_triplets, hypernode_store, new_hypernodes,
                graph, log, ontology=True,
                run_compatibility_check=bool(existing_triplets),
            )
            existing_papers = existing_metadata.get("papers_processed", [])
            if not isinstance(existing_papers, list):
                existing_papers = []
            current_paper = os.path.basename(paper_path)
            if current_paper not in existing_papers:
                existing_papers = existing_papers + [current_paper]
            meta = {
                "last_updated": datetime.datetime.now().isoformat(),
                "papers_processed": existing_papers,
                "version": existing_metadata.get("version", 1),
            }
            save_cookbook_graph(cookbook_file, merged_triplets, merged_hypernodes, meta)
            log(f"Cookbook saved to {cookbook_file} ({len(merged_triplets)} triplets, {len(merged_hypernodes)} hypernodes)")
            # Save expert graph (cookbook) for code generation
            expert_data = {
                "type": "expert",
                "triplets": [[t[0], t[1], t[2]] for t in merged_triplets],
                "hypernode_store": merged_hypernodes,
            }
            log.to_json(expert_data, "graph_data.json")
            log("Expert graph saved to graph_data.json")
        else:
            # No new triplets to merge; save existing cookbook as expert graph
            expert_data = {
                "type": "expert",
                "triplets": [[t[0], t[1], t[2]] for t in existing_triplets],
                "hypernode_store": hypernode_store,
            }
            log.to_json(expert_data, "graph_data.json")
            log("Expert graph saved to graph_data.json")
    
    triplets = graph_data['triplets']
    if not skip_regeneration:
        G = nx.Graph()  # Use Graph for undirected, DiGraph for directed

        for s, o, r in triplets:
            G.add_edge(s, o, label=r['label'])

        #print(f"Total nodes: {G.number_of_nodes()}")
        #print(f"Total edges: {G.number_of_edges()}")

        components = list(nx.connected_components(G))
        #print(f"\nNumber of connected components: {len(components)}")
        # Sort components by size (largest first)
        components_sorted = sorted(components, key=len, reverse=True)

        comps_to_join = set()
        for idx in range(len(components_sorted)):
            component = components_sorted[idx]
            if len(component) > 100:
                subgraph = G.subgraph(component).copy()
                vis_net(subgraph, log_path, save_as=f'comp{idx+1}_{len(component)}nodes')
            else:
                comps_to_join = comps_to_join.union(component)
        if len(comps_to_join) > 0:
            subgraph = G.subgraph(comps_to_join).copy()
            vis_net(subgraph, log_path, save_as=f'small_comps_{len(comps_to_join)}nodes')


    
    # =============================================================================
    # PHASE 3: Generate and validate code from graph
    # =============================================================================
    code_generation = {}
    if generate_code:
        log("="*70)
        log("PHASE 3: GRAPH-ONLY CODE GENERATION")
        log("="*70)
        try:
            code_generation = generate_code_from_graph(
                graph=graph,
                paper_path=paper_path,
                log=log,
                output_dir=log_path,
            )
            #if test_generated_code and code_generation.get("code_path"):
            '''
                exec_results = test_generated_code_executability(
                    code_generation["code_path"], log
                )
                code_generation["executability"] = exec_results
            '''
        except Exception as e:
            log(f"Error in graph-based code generation: {e}")
            code_generation = {"error": str(e)}
        log.to_json(code_generation, "code_generation.json")
        log("code_generation saved to code_generation.json")
    else:
        log("Skipping graph-based code generation (--no-generate-code)")
        
    

    log("="*70)
    log("REPRODUCTION COOKBOOK COMPLETE")
    log("="*70)
    log("")
    

def main():
    parser = argparse.ArgumentParser(
        description='Build reproduction-focused cookbook knowledge graph from academic paper'
    )
    parser.add_argument('--paper', type=str,
                        default='adaptive-pruning',
                        help='Path to paper markdown file')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for retriever (default: cpu)')
    parser.add_argument('--log-path', type=str, default='reproduction_cookbook',
                        help='Path for log output (default: reproduction_cookbook)')
    parser.add_argument('--save-graph', default=True,
                        help='Save final graph structure to JSON')
    parser.add_argument('--generate-code', dest='generate_code', action='store_true',
                        help='Generate code from final graph (default: enabled)')
    parser.add_argument('--no-generate-code', dest='generate_code', action='store_false',
                        help='Disable code generation from graph')
    parser.set_defaults(generate_code=True)
    parser.add_argument('--test-generated-code', dest='test_generated_code', action='store_true',
                        help='Run executability tests for generated code (default: enabled)')
    parser.add_argument('--no-test-generated-code', dest='test_generated_code', action='store_false',
                        help='Disable executability tests for generated code')
    parser.set_defaults(test_generated_code=True)
    parser.add_argument('--repo', type=str, default=None,
                        help='Repository URL for code extraction (overrides blacklist.txt)')
    parser.add_argument('--no-repo', dest='use_repo', action='store_false',
                        help='Skip code processing even if blacklist.txt has repo URL')
    parser.set_defaults(use_repo=True)
    parser.add_argument('--max-code-files', type=int, default=-1,
                        help='Max code files to process (-1 = no limit)')
    parser.add_argument('--cookbook-path', type=str, default=None,
                        help='Path to load/save cookbook JSON (default: ~/arigraph/reproduction_cookbook/cookbook_graph.json)')
    parser.add_argument('--no-load-cookbook', dest='load_cookbook', action='store_false',
                        help='Start with empty graph, do not load existing cookbook')
    parser.set_defaults(load_cookbook=True)
    parser.add_argument('--repo-dir', type=str, default=None,
                        help='Path to local repository (use instead of cloning)')

    args = parser.parse_args()
    starter_path = '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/'
    # '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/short-papers/'
    run_reproduction_test(
        paper_path=starter_path + args.paper + '/paper.md',
        device=args.device,
        log_path=args.log_path + '/' + args.paper,
        generate_code=args.generate_code,
        test_generated_code=args.test_generated_code,
        repo_url_override=args.repo,
        use_repo=args.use_repo,
        max_code_files=args.max_code_files,
        cookbook_path=args.cookbook_path,
        load_cookbook=args.load_cookbook,
        repo_dir=args.repo_dir,
    )


if __name__ == "__main__":
    main()
