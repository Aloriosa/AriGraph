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
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from time import time, sleep
import datetime

from graphs.contriever_graph import ContrieverGraph
from utils.utils import Logger
from tqdm import tqdm

from openai import OpenAI
import networkx as nx
from dotenv import load_dotenv # you'll need to pip install python-dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from prompts.paper_reproduction_prompt import prompt_extraction_reproduction, prompt_extraction_reproduction_with_repo
from prompts.cookbook_extraction_prompt import (
    prompt_cookbook_extraction,
    prompt_cookbook_extraction_wo_code,
    prompt_paper_code_linking,
    prompt_cookbook_reusable_extraction,
    prompt_reusable_pattern_to_code_mapping,
    prompt_coding_agent_paper_graph,
    prompt_coding_agent_paper_mem_graph,
    prompt_pattern_compatibility_check,
    BENCHMARK_EVALUATION_RULES,
    REPRODUCTION_SCRIPT_TOY_EXAMPLE,
)
from graph_storage import (
    load_cookbook_graph,
    save_cookbook_graph,
    merge_triplets_into_cookbook,
    generate_hypernode_id,
)
from ontology.cookbook_ontology import validate_triplet
from utils.retriever_search_drafts import graph_retr_search

import utils_reproduction as reprutil


class ReproductionGraph(ContrieverGraph):
    """Extended ContrieverGraph that uses reproduction-focused prompts."""

    def __init__(self, model, system_prompt, api_key, log, base_url='', device="cpu", type='paper'): # or type 'mem'
        super().__init__(model, system_prompt, api_key, base_url, device)
        if type == 'mem':
            self.graph_builder_instruction = (
                "Extract only REUSABLE patterns for a cumulative cookbook. "
                "Skip paper-specific details. This graph helps implement OTHER papers in the area.\n\n"
            )
            self.reproduction_prompt = prompt_cookbook_reusable_extraction
        elif type == 'paper':
            self.graph_builder_instruction = (
                "Extract all the imprtant information required to reproduce the method described in the paper with all the quantitative and qualitativeresults.\n\n"
            )
            self.reproduction_prompt = prompt_cookbook_extraction_wo_code
        else:
            raise ValueError(f"Invalid type: {type}")
        self.hypernode_store = {}
        self._log = log
        log(f'\nPrompt: {self.reproduction_prompt[:200]}...')
        self.completion_tokens  = 0
        self.prompt_tokens  = 0
        self.triplet2code = {}

    def add_triplets(self, triplets, ontology_validation=False):
        """Override to enforce ontology validation before adding."""
        validated = []
        if ontology_validation:
            for t in triplets:
                if validate_triplet(t):
                    validated.append(t)
                else:
                    self._log(f"Skipping invalid triplet (ontology): {t}")
        else:
            validated = triplets
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
        #self._log(f'\nPrompt: {prompt}...')
        
        response, tokens = self.generate(prompt, t=0.001)
        self.completion_tokens += tokens["completion_tokens"]
        self.prompt_tokens += tokens["prompt_tokens"]
        #log('response generated')
        # Process triplets (reuse parent class logic)
        from utils.utils import process_triplets
        new_triplets_raw = process_triplets(response)
        new_triplets = self.exclude(new_triplets_raw)
        new_triplets_str = self.convert(new_triplets_raw)
        
        self.add_triplets(new_triplets_raw)
        #log('triplets processed')
        
        n_added = len(new_triplets)
        n_parsed = len(new_triplets_raw)
        if n_added < n_parsed:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new (added to graph), {n_parsed - n_added} duplicates or invalid")
        else:
            log(f"Parsed {n_parsed} triplets from {source_type}, {n_added} new")
        
        obs_embedding = self.retriever.embed(observation)
        obs_value = [new_triplets_str, obs_embedding]
        self.obs_episodic[observation] = obs_value
        
        return new_triplets_raw, obs_value



def run_linking_pass_hypernodes(graph, code_index, code_files, log, new_triplets=None):
    """
    Phase 2 linking (hypernode mode): for each code file, use LLM to produce hypernodes
    linking reusable patterns to code. Parses JSON blocks, stores in hypernode_store,
    adds (pattern, implemented_in, hypernode_id) triplets.
    """
    if not new_triplets:
        triplets_to_link = graph.get_all_triplets() if graph else []
    else:
        triplets_to_link = new_triplets
    reusable_patterns_block = "\n".join(f"- {line}" for line in triplets_to_link)

    for rel_path, file_path in code_files:
        if not file_path.exists():
            continue
        content = reprutil.read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 10:
            continue

        rel_path_str = str(rel_path)
        file_index = [e for e in code_index if e["file"] == rel_path_str]
        if not file_index:
            file_index = [{"file": rel_path_str, "type": "file", "name": rel_path_str, "qual": rel_path_str}]

        index_lines = [f"- {e['qual']} ({e['type']})" for e in file_index]
        index_block = "\n".join(index_lines)

        chunks = reprutil.extract_code_chunks(content, rel_path_str)
        code_chunk = "\n\n---\n\n".join(chunks) if chunks else content#[:2000]
        #code_chunk = code_chunk[:4000]
        print(f"inde block: {index_block}\n code chunk: {code_chunk}")
        prompt = prompt_reusable_pattern_to_code_mapping.format(
            reusable_patterns=reusable_patterns_block,
            code_index=index_block,
            file_path=rel_path_str,
            code_chunk=code_chunk,
        )

        try:
            response, _ = graph.generate(prompt)
            blocks = reprutil._extract_json_blocks(response)
            for block in blocks:
                pattern = block.get("pattern")
                hypernode = block.get("hypernode")
                if not pattern or not hypernode:
                    print(f"  Warning: pattern or hypernode not found for {rel_path_str}")
                    continue
                if pattern not in triplets_to_link:
                    continue
                code_val = hypernode.get("code", "")
                doc = hypernode.get("documentation", "")
                imports = hypernode.get("imports", [])
                if not isinstance(imports, list):
                    imports = [imports] if imports else []
                if not code_val and not doc:
                    continue
                if pattern not in graph.triplet2code:
                    log(f"  Hypernode for {pattern} already exists skipping")
                else:
                    graph.triplet2code[pattern] = hypernode
                    log(f"  Hypernode for {pattern} added")
        except Exception as e:
            log(f"  Error linking {rel_path_str}: {e}")
            raise e
    log(f" Unique hypernodes added: {len(graph.triplet2code)}")
    return graph


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

    # Pattern 1: FILE: path (allow any prefix before FILE:, e.g. ##, **, -, 1., etc.)
    file_decl_re = re.compile(r"^\s*.*?FILE:\s*(.+?)\s*\*{0,2}\s*$", re.MULTILINE | re.IGNORECASE)
    # Pattern 2: **/path/to/file** (no FILE: - some LLMs use this format)
    #file_decl_alt_re = re.compile(r"^\s*\*{1,2}\s*(/[a-zA-Z0-9_/.-]+)\s*\*{0,2}\s*$", re.MULTILINE)
    # Fenced code block: ```lang\n...``` (content can span lines)
    code_block_re = re.compile(r"^```(\w*)\s*\n([\s\S]*?)```", re.MULTILINE)
    fence_re = re.compile(r"^```", re.MULTILINE)

    # Find FILE: declarations that are NOT inside fenced code blocks (e.g. markdown
    # content often contains code blocks with "FILE: path" as documentation)
    def _is_inside_code_block(pos):
        # Build (start, end) for each code block; fences can nest (e.g. ```markdown with ```bash inside)
        fences = list(fence_re.finditer(response_text))
        blocks = []
        stack = []
        for m in fences:
            p = m.start()
            rest = response_text[p + 3 :]
            is_opening = rest and (rest[0].isalnum() or rest[0] == "_")
            if is_opening:
                stack.append(p)
            else:
                if stack:
                    start = stack.pop()
                    blocks.append((start, p + 3))
        return any(start <= pos < end for start, end in blocks)

    def _unwrap_markdown_segment(segment):
        """If segment is wrapped in ```markdown ... ```, extract inner content.
        The segment already ends at the next FILE: declaration, so we use the LAST ```
        in the segment as the closing fence - this correctly handles nested code blocks
        (e.g. ```bash, plain ```) inside the markdown."""
        segment = segment.strip()
        if not re.match(r"^```(?:markdown|md)\s*\n", segment, re.IGNORECASE):
            return segment
        start = re.match(r"^```\w*\s*\n", segment, re.IGNORECASE).end()
        # Find the last ``` at start of line - that closes the outer markdown block
        last_fence = -1
        pos = 0
        while True:
            idx = segment.find("```", pos)
            if idx == -1:
                break
            if idx == 0 or segment[idx - 1] == "\n":
                last_fence = idx
            pos = idx + 3
        if last_fence >= 0:
            return segment[start:last_fence].rstrip()
        return segment

    # Collect matches from both patterns, filter out those inside code blocks
    file_matches = [m for m in file_decl_re.finditer(response_text) if not _is_inside_code_block(m.start())]
    #alt_matches = [m for m in file_decl_alt_re.finditer(response_text) if not _is_inside_code_block(m.start())]
    # Merge and sort by position; for alt pattern, use group(1) as path (need to extract path from match)
    #for m in alt_matches:
    #    if not any(fm.start() == m.start() for fm in file_matches):  # avoid duplicate
    #        file_matches.append(m)
    file_matches.sort(key=lambda m: m.start())

    if not file_matches and ("FILE" in response_text.upper() or "**/" in response_text):
        logging.warning(
            "Parse: 'FILE' found in response but no FILE: declarations matched regex; "
            "falling back to single code block or raw text"
        )

    for i, file_match in enumerate(file_matches):
        raw_path = file_match.group(1).strip()
        # Normalize: forward slashes, strip leading slashes
        raw_path = raw_path.replace("\\", "/").lstrip("/")
        # Strip /home/submission/ prefix (common in **/path** format)
        #if raw_path.startswith("home/submission/"):
        #    raw_path = raw_path[len("home/submission/"):]
        #elif raw_path.startswith("/home/submission/"):
        #    raw_path = raw_path[len("/home/submission/"):]
        if not raw_path:
            continue
        # Sanitize: avoid path traversal
        parts = raw_path.split("/")
        safe_parts = [p for p in parts if p and p != ".."]
        rel_path = "/".join(safe_parts) if safe_parts else raw_path
        if not rel_path:
            continue

        start = file_match.end()
        next_file_start = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(response_text)
        segment = response_text[start:next_file_start]

        if rel_path.lower().endswith(".md"):
            content = _unwrap_markdown_segment(segment)
        else:
            code_match = code_block_re.search(segment)
            if not code_match:
                continue
            content = code_match.group(2).rstrip()
        if content.endswith("\n"):
            content = content[:-1]  # Keep trailing newline behavior consistent

        out_path = Path(repo_root) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        #try:
        out_path.write_text(content, encoding="utf-8")
        #except Exception as e:
        #    out_path.write_text(content, encoding="utf-8", errors="replace")
        #    raise e
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
    # remove duplicates
    lines = list(set(lines))
    return "\n".join(lines) if lines else "(none)"


# Implementation-related relations to prioritize when sampling paper queries for retrieval
_IMPL_RELATIONS = frozenset({"implements", "uses", "requires", "configures", "contains", "extends"})


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


def generate_code_from_graph(graph, paper_path, paper_name, log, output_dir=None, max_triplets=None,
                             paper_summary=None, hypernode_store=None, graph_base_dir=None):
    """
    Generate implementation code using paper graph + expert graph.
    Loads paper_graph_data.json and graph_data.json from graph_base_dir (or output_dir if None).
    Writes submission, generated_code_response to output_dir.
    """
    output_dir = graph_base_dir if output_dir is None else output_dir
    graph_data_path = os.path.join(graph_base_dir, "graph_data.json")
    paper_graph_path = os.path.join(graph_base_dir, "paper_graph_data.json")
    
    paper_graph_block = ""
    if os.path.exists(paper_graph_path):
        with open(paper_graph_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)
        paper_triplets = paper_data.get("triplets", [])
        paper_graph_block = _format_triplets_for_prompt(paper_triplets, max_triplets)
    expert_graph_block = False
    if os.path.exists(graph_data_path):
        with open(graph_data_path, "r", encoding="utf-8") as f:
            expert_data = json.load(f)
        expert_triplets = expert_data.get("triplets", [])
        expert_hypernodes = expert_data.get("hypernode_store", {})
        expert_graph_block = _format_expert_graph_for_agent(
            expert_triplets, expert_hypernodes, max_triplets
        )

    if not paper_summary and paper_path:
        
        with open(paper_path, "r", encoding="utf-8") as f:
            full = f.read()
        paper_summary = full[:2000] + ("..." if len(full) > 2000 else "")
        
    if not paper_summary:
        paper_summary = "None"

    
    if paper_graph_block and expert_graph_block:
        prompt = prompt_coding_agent_paper_mem_graph.format(
            BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
            TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
            paper_summary=paper_summary,
            paper_graph=paper_graph_block,
            expert_graph=expert_graph_block,
        )
    elif paper_graph_block:
        prompt = prompt_coding_agent_paper_graph.format(
            BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
            TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
            paper_summary=paper_summary,
            paper_graph=paper_graph_block,
        )
    log(f"Prompt length: {len(prompt)} chars: benchmark rules {len(BENCHMARK_EVALUATION_RULES)} chars, toy example {len(REPRODUCTION_SCRIPT_TOY_EXAMPLE)} chars, paper summary {len(paper_summary)} chars, paper graph {len(paper_graph_block)} chars")
    response, tokens = graph.generate(prompt)
    completion_tokens = tokens["completion_tokens"]
    prompt_tokens = tokens["prompt_tokens"]
    log(f"Code generation response: {len(response)} chars")
    log(f"Code generation tokens: {completion_tokens} completion, {prompt_tokens} prompt")

    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    repo_root = os.path.join(output_dir, paper_name, "submission")
    os.makedirs(repo_root, exist_ok=True)

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    files_created, primary_python = _parse_code_generator_output(response, repo_root)
    code_path = str(primary_python) if primary_python else (os.path.join(repo_root, "generated_reproduction.py") if files_created else None)

    for p in files_created:
        rel = Path(p).relative_to(Path(repo_root))
        log(f"  Created: {rel}")
    log(f"Generated repo at {repo_root} ({len(files_created)} files)")
    log(f"Raw model response saved to: {raw_output_path}")

    return {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created],
        "raw_response_path": raw_output_path,
        "paper_triplets": len(paper_graph_block),
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()),
    }


def generate_code_from_graph_with_retrieval(mem_graph, paper_graph, paper_name, log, output_dir=None, graph_base_dir=None):
    """
    Generate implementation code using paper graph + RETRIEVED relevant snippets from expert graph.
    First uses paper graph to retrieve relevant code snippets from expert knowledge graph,
    then concatenates paper graph + retrieval results into the coding agent input.
    Loads from graph_base_dir (or output_dir if None); writes to output_dir.
    """
    
    agent = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    output_dir = graph_base_dir if output_dir is None else output_dir
    paper_items = {k: 2 for k in paper_graph.get_all_triplets()}

    topk_episodic = 2 
    # subgaph is a list of str triplets
    subgraph, _ = mem_graph.retrieve(paper_items, '', [], topk_episodic)
    log(f"ASSOCIATED SUBGRAPH len: {len(subgraph)}")
    expert_graph = []
    for striplet in subgraph:
        expert_graph.append(f"Pattern: {striplet}, Code snippet: {mem_graph.triplet2code.get(striplet, '')}")

    paper_graph_block = "\n".join(paper_graph.get_all_triplets()) #_format_triplets_for_prompt(paper_graph.get_all_triplets())
    expert_graph_block = "\n".join(expert_graph)
    
    prompt = prompt_coding_agent_paper_mem_graph.format(
        BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
        TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        #paper_summary=paper_summary,
        paper_graph=paper_graph_block,
        expert_graph=expert_graph_block,
    )
    log(f"Prompt length: {len(prompt)} chars, generating code...")
    chat_completion = agent.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a helpful assistant for research reproduction',
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=os.getenv("OPENAI_MODEL"),
                temperature=0.7,
                
            )
    response = chat_completion.choices[0].message.content
    # print(f"Response: {chat_completion}")
    prompt_tokens = chat_completion.usage.prompt_tokens
    completion_tokens = chat_completion.usage.completion_tokens

    log(f"Code generation response: {len(response)} chars")
    log(f"Code generation tokens: {completion_tokens} completion, {prompt_tokens} prompt")

    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, "generated_code_response.txt")
    repo_root = os.path.join(output_dir, paper_name, "submission")
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
        "triplets_used": len(paper_graph.get_all_triplets()) + len(subgraph),
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()),
        "retrieval_variant": True,
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


def link_code_to_mem_graph(mem_graph, log, log_path, paper_path, output_base, repo_dir=None, new_triplets=None):
    log("Linking code to  MEMORY graph...")

    if repo_dir is not None:
        log("="*70)
        log("PHASE 2: CODE INDEX + PAPER-TO-CODE LINKING")
        log("="*70)

        log(f'Using local repo at {repo_dir}')
        try:

            log("\nCollecting code files...")
            code_files = reprutil.collect_code_files(repo_dir)
            log(f"Found {len(code_files)} code files")

            log("\nBuilding code index...")
            code_index = reprutil.build_code_index(repo_dir, code_files, log)
            log(f"Code index: {len(code_index)} entities (classes, functions, config keys)")

            log("\nRunning linking pass (hypernodes: patterns -> code)...")
            link_start = time()
            mem_graph = run_linking_pass_hypernodes(
                mem_graph, code_index, code_files, log, new_triplets=new_triplets
            )
            link_time = time() - link_start
            log(f"Linking complete: {len(mem_graph.triplet2code)} hypernodes in {link_time:.2f}s")
            log(f"Total cost so far: ${mem_graph.total_amount:.4f}")
            log('Saving graph with code linking to a file0')
            reprutil._write_json_to_path(output_base, "mem_graph_data_with_code.json", {
                "paper": os.path.basename(paper_path),
                "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],
                "triplet2code": mem_graph.triplet2code,
                "stats": {"total_triplets": len(mem_graph.triplets), 
                "prompt_tokens": mem_graph.prompt_tokens,
                "completion_tokens": mem_graph.completion_tokens},
            })
            

        except Exception as e:
            log(f"Error in Phase 2: {e}")
            raise e
    else:
        log('No repo provided, skipping Phase 2 (code linking).')
    return mem_graph


def generate_paper_and_mem_graphs(paper_path, output_base, model, api_key, log, base_url, device):
    log("Loading paper...")
    paper_content = reprutil.load_paper(paper_path)
    sections = reprutil.split_into_sections(paper_content)
    log(f"Paper split into {len(sections)} sections")
    log("")

    paper_graph = ReproductionGraph(model, "You are a helpful assistant specializing in research reproduction",
                        api_key, log, base_url, device, type='paper')
    mem_graph = ReproductionGraph(model, "You are a helpful assistant specializing in research replication",
                            api_key, log, base_url, device, type='mem')
    

    # Process sections
    section_stats = []

    for i, section in enumerate(sections):
        section_start_time = time()
        current_datetime = datetime.datetime.now()
        log("="*70)
        log(f"{current_datetime} SECTION {i+1}/{len(sections)}: {section['title']}")
        log("="*70)

        chunks = reprutil.preprocess_section(section['content'])
        log(f"Section split into {len(chunks)} chunks")

        section_triplets_paper = []
        section_triplets_mem = []
        for j, chunk in enumerate(chunks):
            log(f"\n{datetime.datetime.now()} Processing chunk {j+1}/{len(chunks)}...")
            log(f"Chunk preview: {chunk[:100]}...")

            # Pass recent triplets as context for entity consistency across chunks
            #prev_subgraph = _get_context_triplets(paper_graph.triplets, max_triplets=40, recent_first=True)

            retries = 3
            while retries > 0:
                try:
                    log(f"Try {4-retries}/{retries}")
                    new_triplets_paper, _ = paper_graph.update_without_retrieve(chunk, [], log, source_type="paper")
                    section_triplets_paper.extend(new_triplets_paper)
                    new_triplets_mem = None
                    
                    new_triplets_mem, _ = mem_graph.update_without_retrieve(chunk, [], log, source_type="paper")
                    section_triplets_mem.extend(new_triplets_mem)
                    log(f"{datetime.datetime.now()} Chunk {j+1}/{len(chunks)} done")
                    if new_triplets_paper:
                        log("Sample triplets paper:")
                        for triplet in new_triplets_paper[:3]:
                            subj, obj, rel = triplet
                            log(f"  - ({subj}) --[{rel.get('label', 'N/A')}]--> ({obj})")
                    
                    if new_triplets_mem:
                        log("Sample triplets mem:")
                        for triplet in new_triplets_mem[:3]:
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
        log(f"  - Triplets extracted: paper {len(section_triplets_paper)}, mem {len(section_triplets_mem)}")
        log(f"  - Total triplets in graph: paper {len(paper_graph.triplets)}, mem {len(mem_graph.triplets)}")
        log(f"  - Processing time: {section_time:.2f} seconds")
        log(f"  - API tokens: paper {paper_graph.completion_tokens} completion, {paper_graph.prompt_tokens} prompt")
        log(f"  - API tokens: mem {mem_graph.completion_tokens} completion, {mem_graph.prompt_tokens} prompt")
        log("")
        # Save current graph state after each section (to output_base for shared access)
        reprutil._write_json_to_path(output_base, "paper_graph_data.json", {
            "paper": os.path.basename(paper_path),
            "sections_processed": i + 1,
            "triplets": [[t[0], t[1], t[2]] for t in paper_graph.triplets],
            "stats": {"total_triplets": len(paper_graph.triplets), 
            "prompt_tokens": paper_graph.prompt_tokens,
            "completion_tokens": paper_graph.completion_tokens},
        })
        log("Graph state saved to paper_graph_data.json")
        
        reprutil._write_json_to_path(output_base, "mem_graph_data.json", {
            "paper": os.path.basename(paper_path),
            "sections_processed": i + 1,
            "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],
            "stats": {"total_triplets": len(mem_graph.triplets), 
            "prompt_tokens": mem_graph.prompt_tokens,
            "completion_tokens": mem_graph.completion_tokens},
        })
        log("Graph state saved to mem_graph_data.json")

        section_stats.append({
            'section_num': i + 1,
            'title': section['title'],
            'triplets_extracted_paper': len(section_triplets_paper),
            'triplets_extracted_mem': len(section_triplets_mem),
            'processing_time': section_time,
            'triplets_paper': [[t[0], t[1], t[2]] for t in section_triplets_paper],
            'triplets_mem': [[t[0], t[1], t[2]] for t in section_triplets_mem],
        })
    return paper_graph, mem_graph


def generate_code_from_paper_and_mem_graphs(paper_graph, mem_graph, args, log, output_base, output_dir=None, graph_base_dir=None):
            code_generation = {}
            
            log("="*70)
            log("PHASE 3: GRAPH-ONLY CODE GENERATION")
            log("="*70)
            
            code_generation = generate_code_from_graph_with_retrieval(
                mem_graph=mem_graph,
                paper_graph=paper_graph,
                paper_name=args.paper,
                log=log,
                output_dir=os.path.join(output_base, args.experiment) if args.experiment else None,
                graph_base_dir=output_base,
            )
            log.to_json(code_generation, "code_generation.json")
            log("code_generation saved to code_generation.json")
        
            return code_generation


def run_reproduction_test(log, args, paperbench_data_path, n_papers=2):
    """Main function to run reproduction-focused paper test.
    """
    
    for pid, paper_name in enumerate(os.listdir(paperbench_data_path))[:n_papers]:
        paper_path = os.path.join(paperbench_data_path, paper_name + '/paper.md')
        # Experiment path: when set, submission, code_generation, and log go to log_path/experiment_name/
        output_base = args.log_path + '/' + paper_name
        
        
        base_url = os.getenv("OPENAI_API_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL") # "Qwen/Qwen3-Next-80B-A3B-Instruct" #"Qwen/QwQ-32B"

        # Get paper directory and load repo URL
        paper_dir = os.path.dirname(paper_path)
        
        log(f"Paper: {paper_path}")
        log(f"Model: {model}")
        log(f"Device: {args.device}")
        log("")

        total_start_time = time()
        
        paper_graph, add_mem_graph = generate_paper_and_mem_graphs(paper_path, output_base, model, api_key, log, base_url, args.device)
        
        # =============================================================================
        # PHASE 2: Code to mem graph Linking
        # =============================================================================
        if pid == 0:
            mem_graph = add_mem_graph
            mem_graph = link_code_to_mem_graph(mem_graph, log, args.log_path, paper_path, output_base, repo_dir=args.repo_dir)
            # =============================================================================
            # PHASE 3: Generate and validate code from graph
            # =============================================================================
            if args.generate_code:
                code_generation = generate_code_from_paper_and_mem_graphs(paper_graph, mem_graph, args, log, output_base, output_dir=None, graph_base_dir=None)
            else:
                log("Skipping graph-based code generation (--no-generate-code)")
        
        else:
            if args.generate_code:
                # generate code from old mem graph
                code_generation = generate_code_from_paper_and_mem_graphs(paper_graph, mem_graph, args, log, output_dir=None, graph_base_dir=None)
            else:
                log("Skipping graph-based code generation (--no-generate-code)")
            # find new triplets fro add mem graph that are absent in old mem graph
            new_triplets = [t for t in add_mem_graph.triplets if t not in mem_graph.triplets]
            if len(new_triplets) > 0:
                log(f"Found {len(new_triplets)} new triplets to update mem graph")
                # add new triplets to old mem graph
                mem_graph.add_triplets(new_triplets)
                # link code to new mem graph
                mem_graph = link_code_to_mem_graph(mem_graph, log, args.log_path, paper_path, output_base, repo_dir=code_generation["repo_root"], 
                new_triplets=[add_mem_graph.str(t) for t in new_triplets])
                # as triplets in linkuing are sawe and ode files hange the useful ones ay be oerwritten
            else:
                log("No new triplets found to update mem graph, skipping update")
        

        log("="*70)
        log("REPRODUCTION RUN COMPLETE")
        log("="*70)
        log(f"Total time: {time() - total_start_time:.2f} seconds")
        

def main():
    parser = argparse.ArgumentParser(
        description='Build replication-focused cookbook knowledge graph from academic paper'
    )
    parser.add_argument('--paper', type=str,
                        default='adaptive-pruning',
                        help='Path to paper markdown file')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for retriever (default: cpu)')
    parser.add_argument('--log-path', type=str, default='',
                        help='Path for log output (default: none)')
    parser.add_argument('--generate-code', dest='generate_code', default=True, type=bool,
                        help='Generate code from final graph (default: enabled)')
    #parser.add_argument('--test-generated-code', dest='test_generated_code', default=True, type=bool,
    #                    help='Run executability tests for generated code (default: enabled)')
    parser.add_argument('--use-repo', dest='use_repo', default=False, type=bool,
                        help='use repo url from blacklist.txt')
    parser.add_argument('--repo-dir', type=str, default=None,
                        help='Path to local repository (use instead of cloning)')
    parser.add_argument('--code-gen-variant', type=str, default='full',
                        choices=['full', 'retrieval'],
                        help='Code generation variant: full=paper+full expert graph, retrieval=paper+retrieved expert snippets (default: full)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name inside output_dir: creates output_dir/experiment/ with submission, code_generation, and log (default: none)')

    args = parser.parse_args()
    paperbench_data_path = '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/'
    # '/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/short-papers/'
    
    output_base = args.log_path
    if args.experiment:
        experiment_path = os.path.join(output_base, args.experiment)
        os.makedirs(experiment_path, exist_ok=True)
        log = Logger(experiment_path)
        log(f"Experiment: {args.experiment} (output in {experiment_path})")
    else:
        log = Logger(output_base)
        log(f"Output in {output_base}")

    try:
     
        run_reproduction_test(
            log=log,
            args=args,
            paperbench_data_path=paperbench_data_path, 
            n_papers=2
        )

    except Exception as e:
        log("Unexpected Exception: %s" % str(e))
        raise e
        

if __name__ == "__main__":
    main()
