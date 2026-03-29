import re
import json
import os
import subprocess
from pathlib import Path
import ast
from collections import Counter
from pyvis.network import Network
import networkx as nx

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
        font_color="black",#'#10000000' if num_comps < 2 else "black",
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


def _triplet_to_line(triplet):
    """Convert internal triplet representation to prompt-friendly text."""
    subj, obj, rel = triplet
    rel_label = rel.get('label', 'related_to') if isinstance(rel, dict) else str(rel)
    return f"{subj}, {rel_label}, {obj}"


def _write_json_to_path(path, filename, obj):
    """Write JSON to path/filename (used for graph outputs that must stay in output_base)."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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



def collect_code_files(repo_dir):
    """Walk through repository and collect code files."""
    code_files = []
    repo_path = Path(repo_dir)
    
    for file_path in repo_path.rglob('*'):
        if file_path.is_dir():
            continue
        if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRS):
            continue
        
        if file_path.suffix.lower() in CODE_EXTENSIONS:
            try:
                rel_path = file_path.relative_to(repo_path)
                code_files.append((str(rel_path), file_path))
            except Exception as e:
                raise e
    
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
            print(f"File {file_path} does not exist, skipping...")
            continue
        content = read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 10:
            print(f"File {file_path} is binary or too small, skipping...")
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
                    for key in list(data.keys()):
                        index.append({
                            "file": rel_path_str,
                            "type": "config",
                            "name": str(key),
                            "qual": f"{rel_path_str}::{key}",
                        })
            except (json.JSONDecodeError, Exception):
                print(f"Error parsing JSON file {file_path}, skipping...")
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

def _triplet_to_str(triplet):
    """Convert triplet [s, o, {label: r}] to string 's, r, o' for retrieval."""
    if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
        s, o, r = triplet[0], triplet[1], triplet[2]
        rel = r.get("label", r) if isinstance(r, dict) else str(r)
        return f"{s}, {rel}, {o}"
    return ""
