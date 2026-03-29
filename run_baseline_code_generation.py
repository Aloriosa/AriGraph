#!/usr/bin/env python3
"""
Baseline: Take a markdown paper file as input and generate reproduction code
using an LLM agent with the code generation prompt (no knowledge graph).

This serves as a baseline for comparing against the full pipeline that extracts
a knowledge graph from the paper + code before code generation.
"""

import os
import re
import json
import argparse
from pathlib import Path

from graphs.parent_graph import TripletGraph
from utils.utils import Logger
from prompts.cookbook_extraction_prompt import (
    prompt_coding_agent_paper2code,
    BENCHMARK_EVALUATION_RULES,
    REPRODUCTION_SCRIPT_TOY_EXAMPLE,
)
from test_paper_reproduction import load_paper, _parse_code_generator_output
from dotenv import load_dotenv # you'll need to pip install python-dotenv

load_dotenv()
# Same as test_paper_reproduction.py
base_url = os.getenv("OPENAI_API_BASE_URL") # "https://inference.airi.net:46783/v1"
api_key = os.getenv("OPENAI_API_KEY") # "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzMwMDAwNjEsImV4cCI6MTc3MzYwNDg2MX0.xQYUhUzkgTjvFhUplVKuyu0j1T7IEVm-L90NDH1_LLE"
model = os.getenv("OPENAI_MODEL") # "Qwen/Qwen3-Next-80B-A3B-Instruct" #"Qwen/QwQ-32B"



def run_baseline(paper_path, log_path=""):
    """
    Run baseline code generation: paper -> LLM -> code.
    No knowledge graph extraction.
    All output files go to log_path (submission/, generated_code_response.txt, code_generation.json, log.txt).
    """
    paper_path = Path(paper_path).resolve()
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path)

    log("=" * 70)
    log("BASELINE: Paper md -> LLM -> Code")
    log("=" * 70)
    log(f"Paper: {paper_path}")
    log(f"Log path: {log_path}")
    log(f"Model: {model}")
    log("")

    log("Loading paper...")
    paper = load_paper(paper_path)
    log(f"Paper length: {len(paper)} chars")
    log("")

    # Baseline: no paper graph, no expert graph
    
    prompt = prompt_coding_agent_paper2code.format(
        BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
        TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        paper_text=paper,#[:(54000 // 2)],
    )

    log("Initializing LLM...")
    graph = TripletGraph(
        model=model,
        system_prompt="You are a coding agent implementing a research paper for reproduction.",
        api_key=api_key,
        base_url=base_url,
    )

    log("Generating code...")
    completion_tokens = 0
    prompt_tokens = 0
    # log(f"Prompt: {prompt}")
    
    response, tokens = graph.generate(prompt)
    completion_tokens += tokens["completion_tokens"]
    prompt_tokens += tokens["prompt_tokens"]
    #log(f"Response: {response}")
    log(f"API tokens: {completion_tokens} completion, {prompt_tokens} prompt")
    log("")

    repo_root = os.path.join(log_path, "submission")
    os.makedirs(repo_root, exist_ok=True)
    raw_output_path = os.path.join(log_path, "generated_code_response.txt")

    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    files_created, primary_python = _parse_code_generator_output(response, repo_root)
    code_path = str(primary_python) if primary_python else (
        os.path.join(repo_root, "generated_reproduction.py") if files_created else None
    )

    for p in files_created:
        try:
            rel = Path(p).relative_to(Path(repo_root))
        except ValueError:
            rel = p
        log(f"  Created: {rel}")
    log(f"Generated repo at {repo_root} ({len(files_created)} files)")
    log(f"Raw model response saved to: {raw_output_path}")

    result = {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created],
        "raw_response_path": raw_output_path,
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()),
        "api_completion_tokens": completion_tokens,
        "api_prompt_tokens": prompt_tokens,
        "baseline": True,
    }
    log.to_json(result, "code_generation.json")
    log("=" * 70)
    log("BASELINE COMPLETE")
    log("=" * 70)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: generate reproduction code from a markdown paper using LLM (no knowledge graph)"
    )
    parser.add_argument("--paper", type=str, required=True, help="Path to paper markdown file or paper name (e.g. adaptive-pruning)")
    parser.add_argument("--log-path", type=str, default="baseline_output",
                        help="Base path for all output files (default: baseline_output). Output goes to log_path/paper_name/")
    
    args = parser.parse_args()

    
    starter = "/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/"
    paper_path = os.path.join(starter, args.paper, "paper.md")
    

    # Same pattern as test_paper_reproduction: log_path/paper_name
    paper_name = args.paper
    log_path = os.path.join(args.log_path, paper_name)

    run_baseline(
        paper_path=paper_path,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
