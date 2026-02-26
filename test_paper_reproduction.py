#!/usr/bin/env python3
"""
Test script for building a reproduction-focused knowledge graph from academic papers.
Extracts implementation details to create a "cookbook" for reproducing research.
"""

import re
import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from time import time
from collections import Counter
import datetime

from graphs.contriever_graph import ContrieverGraph
from utils.utils import Logger


from pyvis.network import Network
import networkx as nx


sys.path.insert(0, os.path.dirname(__file__))
from prompts.paper_reproduction_prompt import prompt_extraction_reproduction, prompt_extraction_reproduction_with_repo
from prompts.cookbook_extraction_prompt import prompt_cookbook_extraction, prompt_cookbook_extraction_wo_code


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
            "IMPORTANT CONTEXT: Build this knowledge graph for the same agent that will later "
            "implement the paper's code using only this graph as the source of truth, without "
            "access to the original paper. "
            "Prioritize complete, implementation-critical details, explicit dependencies, "
            "ordered procedures, and concrete parameter values whenever available.\n\n"
        )
        self.reproduction_prompt = prompt_cookbook_extraction_wo_code #prompt_cookbook_extraction
        #prompt_extraction_reproduction_with_repo if repo_url else prompt_extraction_reproduction
        log(f'\nPrompt:\n{self.reproduction_prompt}')
    
    def update_without_retrieve(self, observation, prev_subgraph, log, source_type="paper"):
        """Override to use reproduction-focused prompt."""
        example = [re.sub(r"Step \d+: ", "", triplet) for triplet in prev_subgraph]
        
        if source_type == "code":
            observation = f"[CODE IMPLEMENTATION]\n{observation}"
        else:
            observation = f"[PAPER TEXT]\n{observation}"
        
        prompt = self.graph_builder_instruction + self.reproduction_prompt.format(
            observation=observation, 
            example=example
        )
        
        response, _ = self.generate(prompt, t=0.001)
        #log('response generated')
        # Process triplets (reuse parent class logic)
        from utils.utils import process_triplets
        new_triplets_raw = process_triplets(response)
        self.add_triplets(new_triplets_raw)
        #log('triplets processed')
        
        new_triplets = self.exclude(new_triplets_raw)
        log(f"New {len(new_triplets)} triplets from {source_type}")
        
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


def _extract_python_code(text):
    """Extract Python code from fenced output, fallback to raw text."""
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_code_from_graph(graph, paper_path, log, output_dir, max_triplets=1500):
    """
    Generate implementation code from graph triplets only.
    The model receives graph triples as the only technical source of truth.
    """
    unique_lines = []
    seen = set()
    for triplet in graph.triplets:
        line = _triplet_to_line(triplet)
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    graph_lines = unique_lines[:max_triplets]
    graph_block = "\n".join(f"- {line}" for line in graph_lines)

    codegen_prompt = f"""You are an autonomous coding agent.
Your task is to implement the paper using ONLY the knowledge graph below.
You do not have access to the original paper text.

Requirements:
1) Generate a complete, runnable code repository.
2) Include all known hyperparameters/config values from the graph.
3) Follow procedure dependencies and ordering from the graph.
4) If information is missing, add explicit TODO placeholders in code comments.
5) Include core repository files where relevant: `README.md`, `requirements.txt`, config file(s), source modules, training/eval entrypoints, and utility code.
6) Return repository contents in this exact format:
   FILE: path/to/file.ext
   ```language
   <full file content>
   ```
   Repeat for each file, with no extra explanation text.

Paper id: {os.path.basename(paper_path)}
Graph triplets ({len(graph_lines)} shown):
{graph_block}
"""

    response, _ = graph.generate(codegen_prompt, t=0.2)
    generated_code = _extract_python_code(response)

    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    code_path = os.path.join(output_dir, "generated_reproduction.py")

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(generated_code + "\n")

    log(f"Generated code saved to: {code_path}")
    log(f"Raw model response saved to: {raw_output_path}")

    return {
        "code_path": code_path,
        "raw_response_path": raw_output_path,
        "triplets_used": len(graph_lines),
        "code_chars": len(generated_code),
    }


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


def run_reproduction_test(paper_path, device="сpu", log_path="",
max_code_files=-1, generate_code=True, test_generated_code=True):
    """Main function to run reproduction-focused paper test."""
    log = Logger(log_path)
    
    base_url = "https://inference.airi.net:46783/v1"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzIwNjUyNjAsImV4cCI6MTc3MjY3MDA2MH0.AhqeW-tV9jFHrTEFDHauMWIDBVop0Ht8B5UW_y7-fDg"
    model = 'Qwen/QwQ-32B'#'Qwen/Qwen3-Coder-30B-A3B-Instruct'

    # Get paper directory and load repo URL
    paper_dir = os.path.dirname(paper_path)
    repo_url = None #load_repo_url(paper_dir)
    
    log("="*70)
    log("REPRODUCTION COOKBOOK KNOWLEDGE GRAPH BUILDER")
    log("="*70)
    log(f"Paper: {paper_path}")
    log(f"Repository: {repo_url or 'Not specified'}")
    log(f"Model: {model}")
    log(f"Device: {device}")
    log("")

    total_start_time = time()

    log("Initializing reproduction-focused knowledge graph...")
    graph = ReproductionGraph(model, "You are a helpful assistant specializing in research reproduction", 
                             api_key, log, base_url, device)
    log("Graph initialized with reproduction-focused prompt")
    log("")
    
    

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
            log(f"Chunk preview: {chunk[:200]}...")
            
            try:
                new_triplets, _ = graph.update_without_retrieve(chunk, [], log, 
                source_type="paper")
                section_triplets.extend(new_triplets)
                log(f"{datetime.datetime.now()} Extracted {len(new_triplets)}")
                
                if new_triplets:
                    log("Sample triplets:")
                    for triplet in new_triplets[:3]:
                        subj, obj, rel = triplet
                        log(f"  - ({subj}) --[{rel.get('label', 'N/A')}]--> ({obj})")
            except Exception as e:
                log(f"Error processing chunk: {str(e)}")
                assert False
        
        section_time = time() - section_start_time
        
        log("")
        log(f"Section summary:")
        log(f"  - Triplets extracted: {len(section_triplets)}")
        log(f"  - Total triplets in graph: {len(graph.triplets)}")
        log(f"  - Processing time: {section_time:.2f} seconds")
        log(f"  - API cost so far: ${graph.total_amount:.4f}")
        log("")
        
        section_stats.append({
            'section_num': i + 1,
            'title': section['title'],
            'triplets_extracted': len(section_triplets),
            'processing_time': section_time,
            'triplets': [[t[0], t[1], t[2]] for t in section_triplets],
            
        })
    

    
    # =============================================================================
    # PHASE 2: Process Repository Code
    # =============================================================================
    log("="*70)
    log("PHASE 2: PROCESSING REPOSITORY CODE")
    log("="*70)
    
    temp_dir = log_path + '/repo'

    code_stats = []
    if repo_url is not None:
        try:
            if not os.path.exists(temp_dir):
                # Clone repository
                log('Cloning repo...')
                clone_repo(repo_url, temp_dir, log)
            
            else:
                log('Repo is already downloaded, processing it...')
            
            # Collect code files
            log("\nCollecting code files...")
            code_files = collect_code_files(temp_dir, max_files=max_code_files)
            log(f"Found {len(code_files)} important code files to process")
            log("")
            
            code_triplets_total = 0
            
            
            for idx, (rel_path, file_path) in enumerate(code_files):
                log(f"\n{datetime.datetime.now()} Processing file {idx+1}/{len(code_files)}: {rel_path}")
                file_start_time = time()
                file_content = read_file_safe(file_path)
                
                if "[Binary file" in file_content or len(file_content) < 10:
                    log("  Skipping (binary or too small)")
                    continue
                
                file_triplets = []
                # Extract meaningful chunks from code
                chunks = extract_code_chunks(file_content, str(rel_path))
                log(f"  {len(chunks)} code chunks extracted")
                
                for j, chunk in enumerate(chunks):
                    try:
                        # Add file context to chunk
                        chunk_with_context = f"File: {rel_path}\n\n{chunk}"
                        
                        new_triplets, _ = graph.update_without_retrieve(
                            chunk_with_context, [], log, source_type="code"
                        )
                        code_triplets_total += len(new_triplets)
                        file_triplets.extend(new_triplets)
                        
                    except Exception as e:
                        log(f"  Error processing chunk {j+1}: {str(e)}")
                        assert False

                file_time = time() - file_start_time
                code_stats.append({
                'file': str(file_path),
                'triplets_extracted': len(file_triplets),
                'processing_time': file_time,
                'triplets': [[t[0], t[1], t[2]] for t in file_triplets],
                })
                log("")
                log(f"File {idx}/{len(code_files)} processing complete: {len(file_triplets)} triplets extracted")
                log(f"Total cost so far: ${graph.total_amount:.4f}")
                log("")
            
        except subprocess.CalledProcessError as e:
            log(f"Error cloning repository: {e}")
            assert False
        except Exception as e:
            log(f"Error processing repository: {e}")
            assert False
        '''
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log("Cleaned up temporary repository directory")
       '''
    else:
        log('No Repo provided, exiting...')

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
        'stats': {
            'total_triplets': stats['total_triplets'],
            'num_unique_entities': stats['num_unique_entities'],
            'total_time': total_time,
            'total_cost': graph.total_amount,
        },
        'sections': section_stats,
        'code_stats': code_stats,
    }
    log.to_json(graph_data, "graph_data.json")
    log("Graph saved to graph_data.json")
    
    triplets = graph_data['triplets']
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
            if test_generated_code and code_generation.get("code_path"):
                exec_results = test_generated_code_executability(
                    code_generation["code_path"], log
                )
                code_generation["executability"] = exec_results
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
    log("This knowledge graph can guide:")
    log("  1. Setting up the implementation environment")
    log("  2. Configuring hyperparameters correctly")
    log("  3. Understanding the training procedure")
    log("  4. Reproducing experimental results")
    log(f"  5. Navigating the codebase at: {repo_url or 'repository'}")


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
    
    args = parser.parse_args()
    starter_path = '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/'
    # '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/short-papers/'
    run_reproduction_test(
        paper_path= starter_path + args.paper + '/paper.md',
        device=args.device,
        log_path=args.log_path + '/' + args.paper,
        generate_code=args.generate_code,
        test_generated_code=args.test_generated_code,
    )


if __name__ == "__main__":
    main()
