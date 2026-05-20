#!/usr/bin/env python3
"""
Continual code generation across papers (theory / memory graph accumulates).

1. Load the theory (memory) graph from a saved JSON produced after processing the first paper
   (same format as mem_graph_data_with_code.json from test_paper_reproduction.py).
2. For each following paper in order: load that paper's paper_graph_data.json, run topology-aware
   retrieval + code generation, write outputs under {output_root}/{paper_slug}/, then update the
   theory graph in two steps:
   - **Triplet update (from generated code):** LLM extracts reusable triplets from the new
     submission; only triplets that are not already in the graph are added, and an episodic entry
     is stored only when at least one new triplet was produced (no duplicate triplet rows).
   - **Code linking:** Same symbol-index linking as Phase 2, restricted to those new code
     observations and only adding `triplet2code` entries for patterns that did not already have a
     link (no duplicate pattern links; duplicate (pattern, location) pairs in one pass are skipped).
3. Save the cumulative memory graph after each paper for resumability.

Run from the arigraph repo root (same as test_paper_reproduction.py):

  python continual_paper_reproduction.py \\
    --bootstrap-mem-json /home/asagirova/logs/paper1/mem_graph_data_with_code.json \\
    --papers adaptive-pruning semantic-self-consistency lbcs \\
    --paper-graph-base /home/asagirova/logs \\
    --output-root /home/asagirova/arigraph/continual_runs/run_001

Expects {paper_graph_base}/{slug}/paper_graph_data.json for every slug except the first
(the first slug names the bootstrap paper and is not regenerated here).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import time, sleep

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import Logger
import utils_reproduction as reprutil

# Reuse graph class and pipeline stages from the main reproduction test module.
import test_paper_reproduction as tpr

from collections import defaultdict

from utils.utils import clear_triplet, process_triplets


def _smoke_check_generated_code(code_files, log, *, max_fail_ratio: float = 0.2) -> tuple[bool, dict]:
    """Compile every generated .py file. Pass if syntax-error rate <= max_fail_ratio.

    Why: Phase B1/B2 promote patterns from generated code into the cumulative memory graph.
    Without a gate, syntactically broken or hallucinated code propagates as 'reusable patterns'
    to every future paper. A compile check is cheap and catches the most common failure mode
    (LLM emits half-baked code blocks or mid-sentence cutoffs)."""
    py_files = [(rel, p) for rel, p in code_files if p.suffix.lower() in (".py", ".pyw")]
    if not py_files:
        return True, {"py_files": 0, "failures": [], "reason": "no python files"}
    failures: list[tuple[str, str]] = []
    for rel_path, file_path in py_files:
        try:
            source = reprutil.read_file_safe(file_path)
            if "[Binary file" in source:
                continue
            compile(source, str(file_path), "exec")
        except SyntaxError as e:
            failures.append((str(rel_path), f"SyntaxError L{e.lineno}: {e.msg}"))
        except Exception as e:
            failures.append((str(rel_path), f"{type(e).__name__}: {e}"))
    fail_ratio = len(failures) / max(1, len(py_files))
    ok = fail_ratio <= max_fail_ratio
    info = {
        "py_files": len(py_files),
        "failures": failures,
        "fail_ratio": round(fail_ratio, 3),
        "threshold": max_fail_ratio,
    }
    if failures:
        log(f"  Smoke gate: {len(failures)}/{len(py_files)} files failed to compile (ratio {fail_ratio:.2f}, threshold {max_fail_ratio}).")
        for rel, err in failures[:5]:
            log(f"    - {rel}: {err}")
        if len(failures) > 5:
            log(f"    - ... and {len(failures) - 5} more")
    else:
        log(f"  Smoke gate: all {len(py_files)} python files compile cleanly.")
    return ok, info


def _run_paperbench_judge(
    *,
    venv_python: str,
    paperbench_root: str,
    submission_dir: str,
    paper_id: str,
    out_dir: str,
    judge_type: str,
    completer_config: str | None,
    timeout_sec: int,
    log,
) -> tuple[bool, dict]:
    """Subprocess into the paperbench venv to run their LLM judge. Returns (ok, info).

    The judge writes grader_output.json with a top-level `score` in [0, 1] (or None on failure).
    We capture that score; the caller decides the threshold."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        venv_python,
        "-m", "paperbench.scripts.run_judge",
        f"submission_path={submission_dir}",
        f"paper_id={paper_id}",
        f"judge={judge_type}",
        f"out_dir={out_dir}",
    ]
    if completer_config:
        cmd.append(f"completer_config={completer_config}")
    log(f"  Judge: running {' '.join(cmd)}")
    import subprocess
    try:
        proc = subprocess.run(
            cmd, cwd=paperbench_root, capture_output=True, text=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        log(f"  Judge: timed out after {timeout_sec}s")
        return False, {"error": "timeout"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        log(f"  Judge: exit {proc.returncode}: {tail}")
        return False, {"error": "nonzero_exit", "stderr_tail": tail}
    grader_path = os.path.join(out_dir, "grader_output.json")
    if not os.path.isfile(grader_path):
        log(f"  Judge: no grader_output.json at {grader_path}")
        return False, {"error": "no_output"}
    try:
        with open(grader_path, "r", encoding="utf-8") as f:
            grader = json.load(f)
    except Exception as e:
        log(f"  Judge: failed to parse grader_output.json: {e}")
        return False, {"error": f"parse: {e}"}
    score = grader.get("score")
    return True, {"score": score, "grader_output": grader_path}


def distill_memory_graph(
    mem_graph: "tpr.ReproductionGraph",
    log,
    *,
    min_cluster_size: int = 8,
    max_distilled_per_cluster: int = 3,
    cold_storage_path: str | None = None,
) -> dict:
    """Compress unlinked memory triplets via LLM consolidation.

    Without this, memory grows linearly with papers processed and per-paper retrieval cost
    grows with it. Strategy:

      * Asset-linked triplets (keys present in `mem_graph.triplet2code`) are ALWAYS kept —
        they carry the code we'd lose by summarizing.
      * Remaining triplets are bucketed by subject. Buckets >= min_cluster_size are sent
        to the LLM with a 'consolidate into ≤K canonical patterns' instruction (reusing
        the existing extraction prompt — same wire format, same parser).
      * Originals from compressed buckets move to `cold_storage_path` as JSON (recoverable),
        out of the live graph.
      * Observations whose triplet-string list becomes empty after the purge are dropped.

    Returns a dict summary suitable for logging.
    """
    triplet_lines: dict[int, str] = {id(t): mem_graph.str(t) for t in mem_graph.triplets}
    linked_keys = set(mem_graph.triplet2code.keys())

    keep: list = []
    buckets: dict[str, list] = defaultdict(list)
    for t in mem_graph.triplets:
        line = triplet_lines[id(t)]
        if line in linked_keys:
            keep.append(t)
        else:
            buckets[t[0]].append(t)

    distilled_total: list = []
    cold_total: list = []
    seen_keys = {triplet_lines[id(t)] for t in keep}

    for subj, cluster in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(cluster) < min_cluster_size:
            keep.extend(cluster)
            continue
        bullet_lines = [f"- {triplet_lines[id(t)]}" for t in cluster]
        observation = (
            f"[REUSABLE PATTERN CLUSTER for subject '{subj}']\n"
            f"Below are {len(cluster)} previously-extracted triplets about this entity from "
            f"prior papers. Consolidate them into AT MOST {max_distilled_per_cluster} canonical "
            f"reusable patterns: drop near-duplicates, drop paper-specific values, prefer "
            f"general patterns that subsume specific ones. Preserve the (subject, relation, "
            f"object) triplet schema.\n\n" + "\n".join(bullet_lines)
        )
        prompt = mem_graph.graph_builder_instruction + mem_graph.reproduction_prompt.format(
            observation=observation, example=""
        )
        response, tokens = None, None
        last_err = None
        for attempt in range(3):
            try:
                response, tokens = mem_graph.generate(prompt, t=0.001)
                break
            except Exception as e:
                last_err = e
                log(f"  Distill subj={subj!r} attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    from time import sleep
                    sleep(5)
        if response is None:
            log(f"  Distill subj={subj!r}: giving up after 3 retries ({last_err}); keeping originals.")
            keep.extend(cluster)
            continue
        mem_graph.completion_tokens += tokens["completion_tokens"]
        mem_graph.prompt_tokens += tokens["prompt_tokens"]

        parsed = process_triplets(response)
        cluster_distilled: list = []
        for nt in parsed:
            ct = clear_triplet(nt)
            if ct[2].get("label") == "free":
                continue
            line_key = mem_graph.str(ct)
            if line_key in seen_keys:
                continue
            seen_keys.add(line_key)
            cluster_distilled.append(ct)
            if len(cluster_distilled) >= max_distilled_per_cluster:
                break

        if not cluster_distilled:
            log(f"  Distill subj={subj!r}: LLM produced 0 new patterns from {len(cluster)} triplets; keeping originals.")
            keep.extend(cluster)
            continue

        distilled_total.extend(cluster_distilled)
        cold_total.extend(cluster)
        log(f"  Distill subj={subj!r}: {len(cluster)} -> {len(cluster_distilled)} (moved {len(cluster)} to cold)")

    if not cold_total:
        return {"compressed_clusters": 0, "kept": len(mem_graph.triplets), "cold": 0}

    # Rebuild the live triplet list and the embedding cache.
    cold_lines = {mem_graph.str(t) for t in cold_total}
    mem_graph.triplets = list(keep)
    for line in cold_lines:
        mem_graph.triplets_emb.pop(line, None)
    # add_triplets handles embedding for the distilled additions.
    mem_graph.add_triplets(distilled_total)

    # Drop episodic observations whose triplet-string list is now fully cold.
    obs_to_drop: list = []
    obs_updated = 0
    for obs, payload in mem_graph.obs_episodic.items():
        if not isinstance(payload, (list, tuple)) or not payload:
            continue
        lines = payload[0] if payload else []
        remaining = [l for l in lines if l not in cold_lines]
        if not remaining:
            obs_to_drop.append(obs)
        elif len(remaining) < len(lines):
            embedding = payload[1] if len(payload) > 1 else None
            mem_graph.obs_episodic[obs] = [remaining, embedding]
            obs_updated += 1
    for obs in obs_to_drop:
        del mem_graph.obs_episodic[obs]

    # Persist cold storage (append-only JSONL would be ideal; for resumability we use a
    # single JSON with batched runs).
    if cold_storage_path:
        Path(cold_storage_path).parent.mkdir(parents=True, exist_ok=True)
        existing_cold = []
        if os.path.isfile(cold_storage_path):
            try:
                with open(cold_storage_path, "r", encoding="utf-8") as f:
                    existing_cold = json.load(f).get("triplets", [])
            except Exception:
                existing_cold = []
        # Serialize: triplets are [subject, object, {"label": rel}] — JSON-friendly already.
        cold_payload = {
            "triplets": existing_cold + [[t[0], t[1], t[2]] for t in cold_total],
            "last_distill_batch_size": len(cold_total),
        }
        tmp = cold_storage_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cold_payload, f, indent=2)
        os.replace(tmp, cold_storage_path)

    summary = {
        "compressed_clusters": sum(1 for b in buckets.values() if len(b) >= min_cluster_size),
        "kept": len(mem_graph.triplets),
        "cold": len(cold_total),
        "distilled_added": len(distilled_total),
        "obs_updated": obs_updated,
        "obs_dropped": len(obs_to_drop),
    }
    log(
        f"Distillation: compressed {summary['compressed_clusters']} cluster(s); "
        f"live size {summary['kept']} triplets (added {summary['distilled_added']}, "
        f"moved {summary['cold']} to cold); "
        f"obs updated/dropped {summary['obs_updated']}/{summary['obs_dropped']}."
    )
    return summary


def _snapshot_mem_state(mem_graph) -> dict:
    """Cheap snapshot of mem_graph numbers — paired with a later snapshot to derive
    deltas (added triplets, new code links, asset upgrades) per paper."""
    upgrades = 0
    for v in mem_graph.triplet2code.values():
        if isinstance(v, dict):
            upgrades += len(v.get("history", []))
    return {
        "triplets": len(mem_graph.triplets),
        "asset_links": len(mem_graph.triplet2code),
        "upgrade_history_total": upgrades,
        "obs_episodic": len(mem_graph.obs_episodic),
    }


def _safe_read_json(path: str) -> dict | list | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_per_paper_report(
    *,
    slug: str,
    paper_out: str,
    before: dict,
    after: dict,
    t_elapsed: float,
    distill_summary: dict | None,
) -> dict:
    """Tie every per-paper artifact (quality_gate, budget_decision, asset_expansion,
    self_consistency_ranking, retrieval_selection) into a single normalized row so the
    cumulative run_report has everything in one place."""
    qg = _safe_read_json(os.path.join(paper_out, "quality_gate.json")) or {}
    bd = _safe_read_json(os.path.join(paper_out, "budget_decision.json")) or {}
    ax = _safe_read_json(os.path.join(paper_out, "asset_expansion.json")) or {}
    sc = _safe_read_json(os.path.join(paper_out, "self_consistency_ranking.json")) or {}
    rs = _safe_read_json(os.path.join(paper_out, "retrieval_selection.json")) or {}
    pc = _safe_read_json(os.path.join(paper_out, "paper_cards.json")) or {}
    mc = _safe_read_json(os.path.join(paper_out, "mem_cards.json")) or {}

    submission_files = 0
    sub_dir = os.path.join(paper_out, "submission")
    if os.path.isdir(sub_dir):
        for _, _, fns in os.walk(sub_dir):
            submission_files += len(fns)

    judge_info = (qg.get("details") or {}).get("judge") or {}
    smoke_info = (qg.get("details") or {}).get("smoke") or {}

    return {
        "slug": slug,
        "elapsed_sec": round(t_elapsed, 1),
        "graph": {
            "mem_triplets_before": before.get("triplets"),
            "mem_triplets_after": after.get("triplets"),
            "mem_triplets_delta": (after.get("triplets") or 0) - (before.get("triplets") or 0),
            "asset_links_before": before.get("asset_links"),
            "asset_links_after": after.get("asset_links"),
            "asset_links_delta": (after.get("asset_links") or 0) - (before.get("asset_links") or 0),
            "asset_upgrades_this_paper": (after.get("upgrade_history_total") or 0)
                                       - (before.get("upgrade_history_total") or 0),
            "obs_after": after.get("obs_episodic"),
        },
        "retrieval": {
            "paper_cards": len((pc.get("cards") or [])),
            "memory_cards": len((mc.get("cards") or [])),
            "selected_pairs": len((rs.get("selection") or [])),
            "coverage": bd.get("coverage"),
        },
        "budget": (bd.get("tokens") or {}),
        "generation": {
            "n_samples": sc.get("n_samples", 1),
            "winner_idx": sc.get("winner_idx"),
            "winner_score": (
                next((c.get("score") for c in (sc.get("candidates") or []) if c.get("k") == sc.get("winner_idx")), None)
                if sc else None
            ),
            "submission_files": submission_files,
            "asset_expansions": ax.get("total_expansions", 0),
            "asset_char_savings": ax.get("total_char_savings", 0),
            "asset_unknown_ids": len(ax.get("unknown_asset_ids", []) or []),
        },
        "gate": {
            "passed": qg.get("passed"),
            "smoke_fail_ratio": smoke_info.get("fail_ratio"),
            "smoke_failures": len(smoke_info.get("failures", []) or []),
            "judge_score": judge_info.get("score"),
        },
        "distillation": distill_summary,
    }


def _render_run_report_md(rows: list[dict]) -> str:
    if not rows:
        return "# Continual run report\n\n(no papers processed yet)\n"
    lines = [
        "# Continual run report",
        "",
        f"_{len(rows)} paper(s) processed._",
        "",
        "| # | slug | gate | mem Δ | links Δ | upgrades | cov | expansions | savings (≈tok) | files | sec |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    totals = {"savings": 0, "expansions": 0, "upgrades": 0, "links": 0, "secs": 0.0}
    for i, r in enumerate(rows, 1):
        g = r["graph"]; gen = r["generation"]; gate = r["gate"]; ret = r["retrieval"]
        gate_label = "PASS" if gate.get("passed") else ("FAIL" if gate.get("passed") is False else "—")
        cov = ret.get("coverage")
        cov_str = f"{cov:.2f}" if isinstance(cov, (int, float)) else "—"
        savings_tok = (gen.get("asset_char_savings", 0) or 0) // 4
        totals["savings"] += savings_tok
        totals["expansions"] += gen.get("asset_expansions", 0) or 0
        totals["upgrades"] += g.get("asset_upgrades_this_paper", 0) or 0
        totals["links"] += g.get("asset_links_delta", 0) or 0
        totals["secs"] += r.get("elapsed_sec", 0) or 0
        lines.append(
            f"| {i} | `{r['slug']}` | {gate_label} | "
            f"{g.get('mem_triplets_delta', 0):+d} | "
            f"{g.get('asset_links_delta', 0):+d} | "
            f"{g.get('asset_upgrades_this_paper', 0)} | "
            f"{cov_str} | "
            f"{gen.get('asset_expansions', 0)} | "
            f"~{savings_tok} | "
            f"{gen.get('submission_files', 0)} | "
            f"{r.get('elapsed_sec', 0):.0f} |"
        )
    lines.extend([
        "",
        "## Totals",
        f"- Submitted patterns from accepted papers: {totals['links']:+d} new code links, {totals['upgrades']} confidence upgrades",
        f"- Asset-marker reuse across run: {totals['expansions']} expansions ≈ {totals['savings']} output tokens saved",
        f"- Wall-clock: {totals['secs']:.0f}s across {len(rows)} paper(s)",
        "",
    ])
    return "\n".join(lines)


def _update_run_report(output_root: str, row: dict, log) -> None:
    """Append `row` to the cumulative run_report.json and re-render run_report.md."""
    report_path = os.path.join(output_root, "run_report.json")
    existing = _safe_read_json(report_path) or {"papers": []}
    if not isinstance(existing, dict) or "papers" not in existing:
        existing = {"papers": []}
    # Replace any prior row for the same slug (re-run case).
    existing["papers"] = [p for p in existing["papers"] if p.get("slug") != row.get("slug")]
    existing["papers"].append(row)
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, report_path)

    md_path = os.path.join(output_root, "run_report.md")
    md = _render_run_report_md(existing["papers"])
    tmp = md_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(md)
    os.replace(tmp, md_path)
    log(f"Run report updated: {report_path} (+ {md_path})")


def _save_mem_graph(mem_graph, output_path: str, paper_slug: str, log) -> None:
    """Persist memory graph in the same schema as test_paper_reproduction."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper": paper_slug,
        "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],
        "obs_episodic": {k: v[0] for k, v in mem_graph.obs_episodic.items()},
        "triplet2code": mem_graph.triplet2code,
        "stats": {
            "total_triplets": len(mem_graph.triplets),
            "prompt_tokens": mem_graph.prompt_tokens,
            "completion_tokens": mem_graph.completion_tokens,
        },
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, output_path)
    log(f"Saved cumulative memory graph to {output_path}")
    # T4: persist embedding cache so resume / next-paper load skips re-embedding.
    try:
        tpr.save_triplet_embeddings(output_path, mem_graph.triplets_emb, log)
    except Exception as e:
        log(f"T4: failed to save embedding cache (non-fatal): {e}")


def _split_oversized_text(text: str, max_len: int, overlap: int = 400) -> list[str]:
    """Split long text into overlapping windows for LLM extraction."""
    if len(text) <= max_len:
        return [text]
    out = []
    step = max(1, max_len - overlap)
    for i in range(0, len(text), step):
        out.append(text[i : i + max_len])
        if i + max_len >= len(text):
            break
    return out


def append_theory_triplets_from_generated_code_strict(
    mem_graph: tpr.ReproductionGraph,
    repo_root: str,
    log,
    *,
    max_chunk_chars: int = 10_000,
    max_chunks_total: int = 120,
) -> set[str]:
    """
    Extract theory triplets from generated code using the mem graph's code-oriented prompt.
    - Adds only triplets not already present in mem_graph.triplets (graph stays duplicate-free).
    - Skips episodic storage when a chunk yields no new triplets (avoids empty / redundant episodes).
    - Deduplicates within a single LLM parse batch before adding.

    Returns the set of new observation keys added to mem_graph.obs_episodic (for filtered linking).
    """
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        log("Strict code triplet update: no submission directory")
        return set()

    code_files = reprutil.collect_code_files(repo_root)
    new_observation_keys: set[str] = set()
    chunks_processed = 0
    existing_keys: set[str] = {mem_graph.str(t) for t in mem_graph.triplets}

    for rel_path, file_path in code_files:
        if chunks_processed >= max_chunks_total:
            log(f"Strict code triplet update: hit max_chunks_total={max_chunks_total}, stopping.")
            break

        content = reprutil.read_file_safe(file_path)
        if "[Binary file" in content or len(content) < 20:
            continue

        if file_path.suffix.lower() in (".py", ".pyw"):
            pieces = reprutil.extract_code_chunks(content, rel_path, max_chunk_size=min(3000, max_chunk_chars))
            if not pieces:
                pieces = [content]
        else:
            pieces = [content]

        for raw_piece in pieces:
            if chunks_processed >= max_chunks_total:
                break
            for piece in _split_oversized_text(raw_piece, max_chunk_chars):
                if chunks_processed >= max_chunks_total:
                    break
                header = f"### {rel_path}\n"
                observation = f"[CODE IMPLEMENTATION]\n{header}{piece}"

                example_str = ""
                prompt = mem_graph.graph_builder_instruction + mem_graph.reproduction_prompt.format(
                    observation=observation,
                    example=example_str,
                )
                response = None
                tokens = None
                last_err = None
                for attempt in range(3):
                    try:
                        response, tokens = mem_graph.generate(prompt, t=0.001)
                        break
                    except Exception as e:
                        last_err = e
                        log(f"  Code→triplets generate() attempt {attempt + 1}/3 failed: {e}")
                        if attempt < 2:
                            sleep(5)
                if response is None:
                    log(f"  Code→triplets: giving up on chunk {chunks_processed + 1} after 3 retries ({last_err})")
                    chunks_processed += 1
                    continue
                mem_graph.completion_tokens += tokens["completion_tokens"]
                mem_graph.prompt_tokens += tokens["prompt_tokens"]

                parsed = process_triplets(response)
                actually_new: list = []
                # Triplets are lists (unhashable); use canonical string keys for per-batch dedup.
                seen_batch: set[str] = set()
                for t in parsed:
                    ct = clear_triplet(t)
                    if ct[2].get("label") == "free":
                        continue
                    line_key = mem_graph.str(ct)
                    if line_key in existing_keys or line_key in seen_batch:
                        continue
                    seen_batch.add(line_key)
                    actually_new.append(ct)

                if not actually_new:
                    log(
                        f"  Code→triplets chunk {chunks_processed + 1}: 0 new "
                        f"(parsed {len(parsed)}, file {rel_path})"
                    )
                    chunks_processed += 1
                    continue

                mem_graph.add_triplets(actually_new)
                new_lines = [mem_graph.str(t) for t in actually_new]
                existing_keys.update(new_lines)
                # Embedding intentionally None: the continual pipeline only iterates
                # obs_episodic items (no episodic retrieval), and load_triplets_from_json
                # re-embeds on resume if ever needed. Saves one embed per chunk.
                mem_graph.obs_episodic[observation] = [new_lines, None]
                new_observation_keys.add(observation)

                log(
                    f"  Code→triplets chunk {chunks_processed + 1}: +{len(actually_new)} new triplets "
                    f"(parsed {len(parsed)}, file {rel_path})"
                )
                chunks_processed += 1

    log(
        f"Strict code triplet update: {chunks_processed} chunks, "
        f"{len(new_observation_keys)} new episodic observations with new triplets."
    )
    return new_observation_keys


def _build_paper_graph_from_md(
    slug: str,
    paper_md_path: str,
    out_dir: str,
    paper_graph: tpr.ReproductionGraph,
    log,
) -> int:
    """Section-by-section extraction of a paper graph from `paper.md`.

    Mirrors the section loop in `test_paper_reproduction.run_reproduction_test` but only
    builds the paper-type graph (no memory graph, no code linking) so the continual loop
    can self-bootstrap when a paper_graph_data.json is missing.

    Writes paper_graph_data.json to out_dir incrementally (one save per section) for
    crash resilience.
    """
    from time import sleep
    import datetime

    paper_content = reprutil.load_paper(paper_md_path)
    sections = reprutil.split_into_sections(paper_content)
    log(f"  Paper {slug}: extracting graph from {len(sections)} sections of {paper_md_path}")
    out_json = os.path.join(out_dir, "paper_graph_data.json")

    for i, section in enumerate(sections):
        chunks = reprutil.preprocess_section(section["content"])
        log(f"  [{datetime.datetime.now():%H:%M:%S}] section {i+1}/{len(sections)} ({section['title']}): {len(chunks)} chunks")
        for j, chunk in enumerate(chunks):
            for attempt in range(3):
                try:
                    paper_graph.update_without_retrieve(chunk, [], log, source_type="paper")
                    break
                except Exception as e:
                    log(f"    chunk {j+1}/{len(chunks)} attempt {attempt+1}/3 failed: {e}")
                    if attempt < 2:
                        sleep(5)
                    else:
                        raise
        reprutil._write_json_to_path(out_dir, "paper_graph_data.json", {
            "paper": os.path.basename(paper_md_path),
            "sections_processed": i + 1,
            "triplets": [[t[0], t[1], t[2]] for t in paper_graph.triplets],
            "obs_episodic": {k: v[0] for k, v in paper_graph.obs_episodic.items()},
            "stats": {
                "total_triplets": len(paper_graph.triplets),
                "prompt_tokens": paper_graph.prompt_tokens,
                "completion_tokens": paper_graph.completion_tokens,
            },
        })
    log(f"  Built paper graph: {len(paper_graph.triplets)} triplets -> {out_json}")
    return len(paper_graph.triplets)


def _load_paper_graph(
    slug: str,
    paper_graph_json: str,
    model: str,
    api_key: str,
    base_url: str,
    log,
    device: str,
    *,
    paperbench_root: str | None = None,
    auto_build: bool = False,
):
    """Build an empty paper-type graph; populate from disk or extract from paper.md.

    When `auto_build=True` and the JSON is missing, look for `{paperbench_root}/{slug}/paper.md`
    and run the section-by-section extraction inline (writing the JSON to disk for resume).
    """
    paper_graph = tpr.ReproductionGraph(
        model,
        "You are a helpful assistant specializing in research reproduction",
        api_key,
        log,
        base_url,
        device,
        type="paper",
    )
    if not os.path.isfile(paper_graph_json):
        if not auto_build:
            raise FileNotFoundError(
                f"Missing paper graph for {slug}: {paper_graph_json}. "
                f"Re-run with --auto-build-paper-graph (and --paperbench-root) to extract from paper.md."
            )
        if not paperbench_root:
            raise FileNotFoundError(
                f"Missing paper graph for {slug}: {paper_graph_json}. "
                f"--auto-build-paper-graph requires --paperbench-root."
            )
        paper_md = os.path.join(paperbench_root, slug, "paper.md")
        if not os.path.isfile(paper_md):
            raise FileNotFoundError(
                f"Cannot auto-build paper graph for {slug}: paper.md not at {paper_md}"
            )
        os.makedirs(os.path.dirname(paper_graph_json), exist_ok=True)
        _build_paper_graph_from_md(slug, paper_md, os.path.dirname(paper_graph_json), paper_graph, log)
    else:
        paper_graph.load_triplets_from_json(paper_graph_json, clear_first=True)
    log(f"Loaded paper graph for {slug}: {len(paper_graph.triplets)} triplets from {paper_graph_json}")
    return paper_graph


def run_continual(
    *,
    bootstrap_mem_json: str,
    paper_slugs: list[str],
    paper_graph_base: str,
    output_root: str,
    device: str,
    skip_linking: bool,
    skip_code_to_triplets: bool,
    max_chunk_chars: int,
    max_chunks_total: int,
    resume: bool,
    allow_mismatch: bool,
    smoke_gate: bool,
    smoke_max_fail_ratio: float,
    judge_venv_python: str | None,
    judge_paperbench_root: str | None,
    judge_min_score: float,
    judge_type: str,
    judge_completer_config: str | None,
    judge_timeout_sec: int,
    distill_threshold: int,
    distill_min_cluster: int,
    distill_max_out: int,
    replace_confidence_delta: float,
    auto_build_paper_graph: bool,
    paperbench_root: str | None,
    n_samples: int,
    summarize_cards: bool,
    log: Logger,
) -> None:
    if len(paper_slugs) < 1:
        raise ValueError("Need at least one paper slug (the bootstrap / first paper).")

    base_url = os.getenv("OPENAI_API_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")

    os.makedirs(output_root, exist_ok=True)

    mem_graph = tpr.ReproductionGraph(
        model,
        "You are a helpful assistant specializing in research replication",
        api_key,
        log,
        base_url,
        device,
        type="mem",
    )
    resume_seed = None
    completed_slugs: set[str] = set()
    if resume:
        latest_path = os.path.join(output_root, "mem_graph_latest.json")
        if os.path.isfile(latest_path):
            resume_seed = latest_path
        for slug in paper_slugs[1:]:
            if os.path.isfile(os.path.join(output_root, f"mem_graph_after_{slug}.json")):
                completed_slugs.add(slug)

    seed_path = resume_seed or os.path.abspath(bootstrap_mem_json)
    mem_graph.load_triplets_from_json(seed_path, clear_first=True)
    src_label = "resume seed" if resume_seed else "bootstrap"
    log(f"{src_label} theory graph from {seed_path} ({len(mem_graph.triplets)} triplets, {len(mem_graph.triplet2code)} code links)")
    if completed_slugs:
        log(f"Resume: skipping {len(completed_slugs)} already-completed paper(s): {sorted(completed_slugs)}")

    bootstrap_name = Path(bootstrap_mem_json).resolve().parent.name
    if paper_slugs[0] != bootstrap_name:
        msg = (
            f"first slug in --papers ({paper_slugs[0]!r}) differs from bootstrap parent dir name "
            f"({bootstrap_name!r})"
        )
        if allow_mismatch:
            log(f"Note: {msg}; continuing anyway (--allow-mismatch).")
        else:
            raise ValueError(f"{msg}. Re-run with --allow-mismatch to override.")

    # First paper: theory only (already loaded). Papers 2..N: retrieve, generate, link, save.
    for idx in range(1, len(paper_slugs)):
        slug = paper_slugs[idx]
        if slug in completed_slugs:
            log(f"CONTINUAL STEP {idx}/{len(paper_slugs) - 1}: paper {slug!r} — skipped (resume)")
            continue
        t0 = time()
        log("=" * 70)
        log(f"CONTINUAL STEP {idx}/{len(paper_slugs) - 1}: paper {slug!r}")
        log("=" * 70)

        paper_graph_path = os.path.join(paper_graph_base, slug, "paper_graph_data.json")
        paper_out = os.path.join(output_root, slug)
        os.makedirs(paper_out, exist_ok=True)

        mem_state_before = _snapshot_mem_state(mem_graph)
        paper_graph = _load_paper_graph(
            slug, paper_graph_path, model, api_key, base_url, log, device,
            paperbench_root=paperbench_root, auto_build=auto_build_paper_graph,
        )
 
        log("Phase A: retrieval + code generation")
        tpr.generate_code_from_graph_with_retrieval(
            mem_graph=mem_graph,
            paper_graph=paper_graph,
            log=log,
            output_dir=paper_out,
            graph_base_dir=paper_out,
            retrieval_topk=3,
            n_samples=n_samples,
            summarize_cards=summarize_cards,
            card_summary_cache_path=os.path.join(output_root, "card_summary_cache.json"),
        )

        repo_root = os.path.join(paper_out, "submission")
        new_code_obs_keys: set[str] | None = None
        if os.path.isdir(repo_root):
            code_files = reprutil.collect_code_files(repo_root)
            if not code_files:
                log("Phase B: skipped (no code files under submission/)")
            else:
                gate_passed = True
                gate_info: dict = {}
                if smoke_gate:
                    log(f"Quality gate: py_compile smoke check (max fail ratio {smoke_max_fail_ratio})")
                    gate_passed, gate_info["smoke"] = _smoke_check_generated_code(
                        code_files, log, max_fail_ratio=smoke_max_fail_ratio,
                    )
                if gate_passed and judge_venv_python:
                    log(f"Quality gate: paperbench judge ({judge_type}, min score {judge_min_score})")
                    judge_ok, judge_info = _run_paperbench_judge(
                        venv_python=judge_venv_python,
                        paperbench_root=judge_paperbench_root or "",
                        submission_dir=repo_root,
                        paper_id=slug,
                        out_dir=os.path.join(paper_out, "judge"),
                        judge_type=judge_type,
                        completer_config=judge_completer_config,
                        timeout_sec=judge_timeout_sec,
                        log=log,
                    )
                    gate_info["judge"] = judge_info
                    if not judge_ok:
                        gate_passed = False
                    else:
                        score = judge_info.get("score")
                        if score is None or score < judge_min_score:
                            log(f"  Judge: score {score} < {judge_min_score}, gate failed.")
                            gate_passed = False
                        else:
                            log(f"  Judge: score {score} >= {judge_min_score}, gate passed.")
                # Always persist the gate decision for postmortem
                reprutil._write_json_to_path(paper_out, "quality_gate.json", {
                    "passed": gate_passed, "details": gate_info,
                })
                if not gate_passed:
                    log(
                        "Phase B: SKIPPED — quality gate failed. Memory graph is NOT updated "
                        "from this paper's generated code (avoids polluting future papers)."
                    )
                    new_code_obs_keys = set()
                elif skip_code_to_triplets:
                    log("Phase B1: skipped (--skip-code-to-triplets); Phase B2 will also be skipped (no new obs)")
                    new_code_obs_keys = set()
                else:
                    log(f"Phase B1: theory triplets from new code ({len(code_files)} files, strict dedup)")
                    new_code_obs_keys = append_theory_triplets_from_generated_code_strict(
                        mem_graph,
                        repo_root,
                        log,
                        max_chunk_chars=max_chunk_chars,
                        max_chunks_total=max_chunks_total,
                    )

                if skip_linking:
                    log("Phase B2: skipped (--skip-linking)")
                elif new_code_obs_keys is not None and len(new_code_obs_keys) == 0:
                    log(
                        "Phase B2: skipped (Phase B1 added no new triplets / observations for this paper). "
                        "Use --skip-code-to-triplets to run linking on all episodic observations instead."
                    )
                else:
                    if new_code_obs_keys is None:
                        log(
                            "Phase B2: code linking on all episodic observations "
                            "(only_new_patterns; no duplicate triplet2code keys; duplicate pattern+location pairs skipped)"
                        )
                    else:
                        log(
                            f"Phase B2: code linking on {len(new_code_obs_keys)} new code observation(s) only "
                            "(only_new_patterns; no duplicate triplet2code keys)"
                        )
                    tpr.run_linking_pass_hypernodes(
                        mem_graph,
                        code_files,
                        log,
                        output_base=paper_out,
                        paper_path=paper_graph_path,
                        repo_dir=repo_root,
                        observation_keys_filter=new_code_obs_keys,
                        only_new_patterns=True,
                        replace_confidence_delta=replace_confidence_delta,
                    )
        else:
            log("Phase B: skipped (submission directory missing)")

        # Optional: bound live memory by distilling redundant non-asset-linked clusters.
        distill_summary: dict | None = None
        if distill_threshold > 0 and len(mem_graph.triplets) > distill_threshold:
            log(
                f"Memory distillation triggered: live size {len(mem_graph.triplets)} > "
                f"threshold {distill_threshold}"
            )
            distill_summary = distill_memory_graph(
                mem_graph,
                log,
                min_cluster_size=distill_min_cluster,
                max_distilled_per_cluster=distill_max_out,
                cold_storage_path=os.path.join(output_root, "mem_graph_cold.json"),
            )

        cumulative_path = os.path.join(output_root, f"mem_graph_after_{slug}.json")
        _save_mem_graph(mem_graph, cumulative_path, slug, log)
        _save_mem_graph(mem_graph, os.path.join(output_root, "mem_graph_latest.json"), slug, log)

        elapsed = time() - t0
        mem_state_after = _snapshot_mem_state(mem_graph)
        try:
            row = _build_per_paper_report(
                slug=slug,
                paper_out=paper_out,
                before=mem_state_before,
                after=mem_state_after,
                t_elapsed=elapsed,
                distill_summary=distill_summary,
            )
            _update_run_report(output_root, row, log)
        except Exception as e:
            log(f"Run report update failed (non-fatal): {e}")

        log(f"Paper {slug} done in {elapsed:.1f}s; outputs under {paper_out}")

    log("=" * 70)
    log("CONTINUAL RUN COMPLETE")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Continual retrieval+code generation: grow theory graph across papers (see module docstring)."
    )
    parser.add_argument(
        "--bootstrap-mem-json",
        type=str,
        required=True,
        help="Path to mem_graph_data_with_code.json from the first paper (theory graph seed).",
    )
    parser.add_argument(
        "--papers",
        type=str,
        nargs="+",
        required=True,
        help="Ordered paper slugs: [first, second, ...]. First paper seeds theory from --bootstrap-mem-json only; "
        "from the second slug onward, paper_graph_data.json must exist under --paper-graph-base/{slug}/.",
    )
    parser.add_argument(
        "--paper-graph-base",
        type=str,
        required=True,
        help="Directory containing {slug}/paper_graph_data.json for continuation papers.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="All outputs under /home/asagirova/...: per-paper folders + mem_graph_latest.json",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--skip-linking",
        action="store_true",
        help="Skip Phase B2 (code linking). Phase B1 (new triplets from code) still runs unless --skip-code-to-triplets.",
    )
    parser.add_argument(
        "--skip-code-to-triplets",
        action="store_true",
        help="Skip Phase B1 (LLM triplet extraction from generated code). If set, linking is skipped unless new obs exist from elsewhere.",
    )
    parser.add_argument(
        "--max-code-chunk-chars",
        type=int,
        default=10_000,
        help="Max characters per code chunk for triplet extraction (default 10000).",
    )
    parser.add_argument(
        "--max-code-chunks-total",
        type=int,
        default=120,
        help="Safety cap on code chunks processed per paper for triplet extraction (default 120).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: load mem_graph_latest.json from --output-root (if present) instead of --bootstrap-mem-json, "
        "and skip any slug whose mem_graph_after_{slug}.json already exists.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow the first slug in --papers to differ from the bootstrap JSON's parent directory name "
        "(otherwise this is a hard error to catch typos).",
    )
    # Quality gates (Phase B1/B2 are skipped if any enabled gate fails, so broken
    # generated code does not pollute the cumulative memory graph).
    parser.add_argument(
        "--no-smoke-gate", dest="smoke_gate", action="store_false",
        help="Disable the py_compile smoke check (default: enabled). Without this gate, syntactically "
        "broken generated code can be promoted into memory and propagate to future papers.",
    )
    parser.add_argument(
        "--smoke-max-fail-ratio", type=float, default=0.2,
        help="Maximum allowed ratio of .py files failing to compile (default 0.2).",
    )
    parser.add_argument(
        "--judge-venv-python", type=str, default=None,
        help="Optional: absolute path to the paperbench venv's python. When set, runs the paperbench "
        "LLM judge against the generated submission and gates Phase B on the score.",
    )
    parser.add_argument(
        "--judge-paperbench-root", type=str, default=None,
        help="Working directory for the paperbench judge subprocess (its repo root). "
        "Required when --judge-venv-python is set.",
    )
    parser.add_argument(
        "--judge-min-score", type=float, default=0.3,
        help="Minimum paperbench score in [0,1] required to pass the judge gate (default 0.3).",
    )
    parser.add_argument(
        "--judge-type", type=str, default="simple",
        choices=["simple", "dummy", "random"],
        help="paperbench judge type (default 'simple').",
    )
    parser.add_argument(
        "--judge-completer-config", type=str, default=None,
        help="paperbench completer_config string (required for --judge-type=simple).",
    )
    parser.add_argument(
        "--judge-timeout-sec", type=int, default=1800,
        help="Timeout for the judge subprocess in seconds (default 1800).",
    )
    # Memory distillation (bounds live graph size across many papers).
    parser.add_argument(
        "--distill-threshold", type=int, default=0,
        help="When >0, after each paper run distillation if live mem_graph triplets exceed this. "
        "Default 0 = disabled. Suggested value: 800-1500 for long continual runs.",
    )
    parser.add_argument(
        "--distill-min-cluster", type=int, default=8,
        help="Only clusters (grouped by triplet subject) with at least this many non-asset-linked "
        "triplets are sent to the LLM for consolidation (default 8).",
    )
    parser.add_argument(
        "--distill-max-out", type=int, default=3,
        help="Maximum canonical patterns kept per compressed cluster (default 3).",
    )
    parser.add_argument(
        "--replace-confidence-delta", type=float, default=0.0,
        help="When > 0, an existing triplet2code link can be replaced by a new one whose "
        "confidence beats it by at least this margin (the old entry is preserved in "
        "`history`). Default 0 = locked-in (paper-1 code never replaced by later papers). "
        "Suggested value: 0.15.",
    )
    parser.add_argument(
        "--auto-build-paper-graph", action="store_true",
        help="If a paper's paper_graph_data.json is missing under --paper-graph-base/{slug}/, "
        "extract it inline from {paperbench-root}/{slug}/paper.md instead of failing.",
    )
    parser.add_argument(
        "--paperbench-root", type=str, default=None,
        help="Root with {slug}/paper.md (required when --auto-build-paper-graph is set).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1,
        help="Self-consistency: number of generation samples per paper. N>1 generates N candidates "
        "at t=0.7, scores by compile_ratio + asset-marker reuse + size, keeps the best (Q5). "
        "Cost scales linearly with N. Default 1 = current behavior.",
    )
    parser.add_argument(
        "--summarize-cards", action="store_true",
        help="Q1: compress each memory card's triplet list into a one-line LLM summary "
        "before prompt packing (cluster size >= 3 only). One short LLM call per new cluster, "
        "cached across papers via {output_root}/card_summary_cache.json. Reduces memory-card "
        "token footprint substantially once the cache warms.",
    )
    args = parser.parse_args()

    out = os.path.abspath(args.output_root)
    os.makedirs(out, exist_ok=True)
    log = Logger(out)

    try:
        run_continual(
            bootstrap_mem_json=args.bootstrap_mem_json,
            paper_slugs=list(args.papers),
            paper_graph_base=os.path.abspath(args.paper_graph_base),
            output_root=out,
            device=args.device,
            skip_linking=args.skip_linking,
            skip_code_to_triplets=args.skip_code_to_triplets,
            max_chunk_chars=args.max_code_chunk_chars,
            max_chunks_total=args.max_code_chunks_total,
            resume=args.resume,
            allow_mismatch=args.allow_mismatch,
            smoke_gate=args.smoke_gate,
            smoke_max_fail_ratio=args.smoke_max_fail_ratio,
            judge_venv_python=args.judge_venv_python,
            judge_paperbench_root=args.judge_paperbench_root,
            judge_min_score=args.judge_min_score,
            judge_type=args.judge_type,
            judge_completer_config=args.judge_completer_config,
            judge_timeout_sec=args.judge_timeout_sec,
            distill_threshold=args.distill_threshold,
            distill_min_cluster=args.distill_min_cluster,
            distill_max_out=args.distill_max_out,
            replace_confidence_delta=args.replace_confidence_delta,
            auto_build_paper_graph=args.auto_build_paper_graph,
            paperbench_root=(os.path.abspath(args.paperbench_root) if args.paperbench_root else None),
            n_samples=max(1, args.n_samples),
            summarize_cards=args.summarize_cards,
            log=log,
        )
    except Exception as e:
        log(f"Continual run failed: {e}")
        raise


if __name__ == "__main__":
    main()
