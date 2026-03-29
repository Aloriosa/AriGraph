#!/usr/bin/env python3
"""
ReAct Baseline: Iteratively improve generated code based on the paper markdown.

Uses a ReAct (Reasoning + Acting) loop:
1. **Act**: Generate initial code from paper
2. **Observe**: Run compile check, optionally execute reproduce.sh
3. **Reason**: Analyze errors/feedback against the paper
4. **Act**: Generate improved code addressing the feedback
5. Repeat until max iterations or success

This serves as a baseline that iteratively refines code using the paper as ground truth.
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

from graphs.parent_graph import TripletGraph
from utils.utils import Logger
from prompts.cookbook_extraction_prompt import (
    prompt_coding_agent_paper2code,
    prompt_react_refinement,
    prompt_paper_reproduction_check,
    BENCHMARK_EVALUATION_RULES,
    REPRODUCTION_SCRIPT_TOY_EXAMPLE,
)
from utils_reproduction import load_paper, read_file_safe
from test_paper_reproduction import _parse_code_generator_output
from dotenv import load_dotenv # you'll need to pip install python-dotenv

load_dotenv()
# Same as run_baseline_code_generation.py
base_url = os.getenv("OPENAI_API_BASE_URL") # "https://inference.airi.net:46783/v1"
api_key = os.getenv("OPENAI_API_KEY") # "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzMwMDAwNjEsImV4cCI6MTc3MzYwNDg2MX0.xQYUhUzkgTjvFhUplVKuyu0j1T7IEVm-L90NDH1_LLE"
model = os.getenv("OPENAI_MODEL") # "Qwen/Qwen3-Next-80B-A3B-Instruct" #"Qwen/QwQ-32B"



def _gather_current_code(repo_root):
    """Gather all code files from repo_root into a single string for the prompt."""
    parts = []
    repo_path = Path(repo_root)
    if not repo_path.exists():
        return "(no files)"
    for f in sorted(repo_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".py", ".sh", ".md", ".yaml", ".yml", ".json", ".txt"):
            try:
                content = read_file_safe(f)
                rel = f.relative_to(repo_path)
                parts.append(f"--- {rel} ---\n{content}")
            except Exception:
                pass
    return "\n\n".join(parts) if parts else "(no files)"


def _run_compile_check(repo_root, code_path=None, timeout_sec=30):
    """Run py_compile on Python files. Returns (success, stderr)."""
    repo_path = Path(repo_root)
    if not repo_path.exists():
        return False, "Repo not found"
    py_files = list(repo_path.rglob("*.py"))
    if not py_files:
        return False, "No Python files to check"
    # Prefer primary, else check all
    to_check = [Path(code_path)] if code_path and Path(code_path).exists() else py_files
    to_check = [p for p in to_check if p.exists()]
    if not to_check:
        to_check = py_files
    errors = []
    for p in to_check:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(p)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(repo_root),
            )
            if result.returncode != 0:
                errors.append(f"{p.name}: {result.stderr or result.stdout or 'compile failed'}")
        except subprocess.TimeoutExpired:
            errors.append(f"{p.name}: timeout")
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    return len(errors) == 0, "\n".join(errors) if errors else ""


def _run_llm_paper_reproduction_check(graph, paper_text, generated_code, paper_max_chars=15000, code_max_chars=25000):
    """
    LLM-based check: compare paper vs code, return feedback on missing/incorrect implementations.
    Returns (all_ok: bool, feedback: str, tokens: dict).
    """
    prompt = prompt_paper_reproduction_check.format(
        paper_text=paper_text[:paper_max_chars],
        generated_code=generated_code[:code_max_chars],
    )
    response, tokens = graph.generate(prompt, t=0.1)
    response = response.strip()
    all_ok = response.upper().startswith("ALL_OK:")
    return all_ok, response, tokens


def _run_reproduce_sh(repo_root, timeout_sec=120):
    """Run reproduce.sh if it exists. Returns (success, stdout, stderr)."""
    script = Path(repo_root) / "reproduce.sh"
    if not script.exists():
        return False, "", "reproduce.sh not found"
    try:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = (result.stdout or "")[-4000:]
        stderr = (result.stderr or "")[-4000:]
        return result.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired as e:
        return False, (e.stdout or "")[-4000:], f"Timeout after {timeout_sec}s"
    except Exception as e:
        return False, "", str(e)


def run_react_baseline(
    paper_path,
    log_path="",
    max_iterations=5,
    run_reproduce=False,
    reproduce_timeout=120,
    run_llm_check=True,
):
    """
    Run ReAct baseline: paper -> generate -> observe -> refine (loop).

    max_iterations: max refinement rounds (1 = no refinement, 2+ = iterative improvement)
    run_reproduce: if True, run reproduce.sh as observation (slower, more feedback)
    run_llm_check: if True, use LLM to verify paper requirements are reproduced in code
    """
    paper_path = Path(paper_path).resolve()
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path)
    trial_num = int(log_path.split('_trial')[-1][0])
    log(f"Trial number: {trial_num}")
    log("=" * 70)
    log("REACT BASELINE: Paper md -> Generate -> Observe -> Refine (loop)")
    log("=" * 70)
    log(f"Paper: {paper_path}")
    log(f"Log path: {log_path}")
    log(f"Model: {model}")
    log(f"Max iterations: {max_iterations}")
    log(f"Run reproduce.sh: {run_reproduce}")
    log(f"Run LLM paper check: {run_llm_check}")
    log("")

    log("Loading paper...")
    paper = load_paper(paper_path)
    paper_truncated = paper#[: (54000 // 2)]
    log(f"Paper length: {len(paper)} chars (using {len(paper_truncated)} for prompt)")
    log("")

    graph = TripletGraph(
        model=model,
        system_prompt="You are a coding agent implementing a research paper. You reason about the paper and iteratively improve your code based on feedback.",
        api_key=api_key,
        base_url=base_url,
    )

    repo_root = os.path.join(os.getcwd(), log_path, f"submission_trial{trial_num}")
    os.makedirs(repo_root, exist_ok=True)

    total_completion_tokens = 0
    total_prompt_tokens = 0
    iteration = 0
    feedback_history = []
    last_response = None

    # Initial generation
    log("--- Iteration 0: Initial generation ---")
    prompt = prompt_coding_agent_paper2code.format(
        BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
        TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
        paper_text=paper_truncated,
    )
    response, tokens = graph.generate(prompt)
    total_completion_tokens += tokens["completion_tokens"]
    total_prompt_tokens += tokens["prompt_tokens"]
    last_response = response

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
    log("")

    # ReAct loop: observe -> reason -> refine
    for iteration in range(1, max_iterations + 1):
        log(f"--- Iteration {iteration}: Observe & Refine ---")

        feedback_parts = []

        # Observe: compile check
        compile_ok, compile_err = _run_compile_check(repo_root, code_path)
        if not compile_ok:
            feedback_parts.append(f"COMPILE ERROR:\n{compile_err}")
        else:
            feedback_parts.append("Compile check: PASSED")

        # Observe: run reproduce.sh (optional)
        if run_reproduce:
            run_ok, stdout, stderr = _run_reproduce_sh(repo_root, timeout_sec=reproduce_timeout)
            if not run_ok:
                feedback_parts.append(f"REPRODUCE.SH FAILED:\nstdout:\n{stdout}\nstderr:\n{stderr}")
            else:
                feedback_parts.append("reproduce.sh: PASSED")

        # Observe: LLM-based paper reproduction check
        current_code = _gather_current_code(repo_root)
        if run_llm_check:
            llm_ok, llm_feedback, llm_tokens = _run_llm_paper_reproduction_check(
                graph, paper_truncated, current_code
            )
            total_completion_tokens += llm_tokens.get("completion_tokens", 0)
            total_prompt_tokens += llm_tokens.get("prompt_tokens", 0)
            if not llm_ok:
                feedback_parts.append(f"PAPER REPRODUCTION CHECK (missing/incorrect):\n{llm_feedback}")
            else:
                feedback_parts.append("Paper reproduction check: PASSED")

        feedback = "\n\n".join(feedback_parts)
        feedback_history.append(feedback)

        # If all checks passed, we can stop early
        failure_indicators = [
            "COMPILE ERROR",
            "REPRODUCE.SH FAILED",
            "PAPER REPRODUCTION CHECK (missing/incorrect)",
        ]
        if not any(ind in feedback for ind in failure_indicators):
            log("All checks passed. Stopping refinement.")
            break

        log(f"Feedback:\n{feedback}")
        log("")

        # Reason + Act: refine code
        prompt = prompt_react_refinement.format(
            BENCHMARK_RULES=BENCHMARK_EVALUATION_RULES,
            TOY_EXAMPLE=REPRODUCTION_SCRIPT_TOY_EXAMPLE,
            paper_text=paper_truncated,#[:2000],
            current_code=current_code,#[:15000],
            feedback=feedback,#[:2000],
        )

        log("Generating refined code...")
        response, tokens = graph.generate(prompt)
        total_completion_tokens += tokens["completion_tokens"]
        total_prompt_tokens += tokens["prompt_tokens"]
        last_response = response

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
            log(f"  Updated: {rel}")
        log("")

    log("=" * 70)
    log("REACT BASELINE COMPLETE")
    log("=" * 70)
    log(f"Total iterations: {iteration + 1}")
    log(f"API tokens: {total_completion_tokens} completion, {total_prompt_tokens} prompt")
    log(f"Raw model response saved to: {raw_output_path}")

    result = {
        "code_path": code_path,
        "repo_root": repo_root,
        "files_created": [str(p) for p in files_created] if files_created else [],
        "raw_response_path": raw_output_path,
        "code_chars": sum(p.stat().st_size for p in files_created if p.exists()) if files_created else 0,
        "api_completion_tokens": total_completion_tokens,
        "api_prompt_tokens": total_prompt_tokens,
        "iterations": iteration + 1,
        "feedback_history": feedback_history,
        "react_baseline": True,
    }
    log.to_json(result, "code_generation.json")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="ReAct baseline: iteratively improve reproduction code from paper using observe-refine loop"
    )
    parser.add_argument("--paper", type=str, required=True, help="Path to paper markdown file or paper name (e.g. adaptive-pruning)")
    parser.add_argument("--log-path", type=str, default="react_baseline_output",
                        help="Base path for all output files (default: react_baseline_output)")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max refinement iterations (default: 5)")
    parser.add_argument("--run-reproduce", action="store_true",
                        help="Run reproduce.sh each iteration for richer feedback (slower)")
    parser.add_argument("--reproduce-timeout", type=int, default=120,
                        help="Timeout in seconds for reproduce.sh (default: 120)")
    parser.add_argument("--no-llm-check", dest="run_llm_check", action="store_false",
                        help="Disable LLM-based paper reproduction check")
    parser.set_defaults(run_llm_check=True)

    args = parser.parse_args()

    starter = "/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers/"
    paper_path = os.path.join(starter, args.paper, "paper.md")

    paper_name = args.paper
    log_path = os.path.join(args.log_path, paper_name)

    run_react_baseline(
        paper_path=paper_path,
        log_path=log_path,
        max_iterations=args.max_iterations,
        run_reproduce=args.run_reproduce,
        reproduce_timeout=args.reproduce_timeout,
        run_llm_check=args.run_llm_check,
    )


if __name__ == "__main__":
    main()
