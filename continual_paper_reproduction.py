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
    --paperbench-root /home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers \\
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
from time import time

from dotenv import load_dotenv
from openai import APITimeoutError

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import Logger
import utils_reproduction as reprutil

# Reuse graph class and pipeline stages from the main reproduction test module.
import test_paper_reproduction as tpr

from utils.utils import clear_triplet, process_triplets


def _save_mem_graph(mem_graph, output_path: str, paper_slug: str, log) -> None:
    """Persist memory graph in the same schema as test_paper_reproduction."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper": paper_slug,
        "triplets": [[t[0], t[1], t[2]] for t in mem_graph.triplets],
        "obs_episodic": {k: v[0] for k, v in mem_graph.obs_episodic.items()},
        "triplet2code": mem_graph.triplet2code,
        "triplet_origin": getattr(mem_graph, "triplet_origin", {}),
        "stats": {
            "total_triplets": len(mem_graph.triplets),
            "prompt_tokens": mem_graph.prompt_tokens,
            "completion_tokens": mem_graph.completion_tokens,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"Saved cumulative memory graph to {output_path}")

    # T4: persist embedding cache so resume / next-paper load skips re-embedding.
    try:
        tpr.save_graph_embeddings(output_path, mem_graph, log)
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
    current_slug: str,
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
                # On timeout, skip this chunk and move on (no retry, no crash).
                try:
                    response, tokens = mem_graph.generate(prompt, t=0.001)
                except APITimeoutError:
                    log(f"  Code→triplets chunk {chunks_processed + 1}: timed out, skipping (file {rel_path})")
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
                    if ct in mem_graph.triplets or line_key in seen_batch:
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
                if not hasattr(mem_graph, "triplet_origin"):
                    mem_graph.triplet_origin = {}
                for t in actually_new:
                    mem_graph.triplet_origin.setdefault(mem_graph.str(t), current_slug)
                new_lines = [mem_graph.str(t) for t in actually_new]
                obs_embedding = mem_graph.retriever.embed(observation)
                mem_graph.obs_episodic[observation] = [new_lines, obs_embedding]
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


def _log_retrieval_origins(mem_graph, paper_out: str, current_slug: str, log) -> None:
    """Log originating paper for each retrieved mem card and persist enrichment.

    Reads retrieval_selection.json + mem_graph_typed.json that
    generate_code_from_graph_with_retrieval just wrote into paper_out.
    Resolves each selected memory card's member triplets back to the raw
    mem_graph triplet (by `triplet_{i:05d}` index), then to the origin slug
    via `mem_graph.triplet_origin`. Rewrites retrieval_selection.json with
    an `origin_papers` field per row.
    """
    sel_path = os.path.join(paper_out, "retrieval_selection.json")
    typed_path = os.path.join(paper_out, "mem_graph_typed.json")
    cards_path = os.path.join(paper_out, "mem_cards.json")
    if not (os.path.isfile(sel_path) and os.path.isfile(typed_path) and os.path.isfile(cards_path)):
        log("Retrieval origins: artifact(s) missing, skipping origin annotation.")
        return

    try:
        with open(sel_path, "r", encoding="utf-8") as f:
            sel_data = json.load(f)
        with open(cards_path, "r", encoding="utf-8") as f:
            cards_data = json.load(f)
    except Exception as e:
        log(f"Retrieval origins: failed to read artifacts ({e})")
        return

    origin_map = getattr(mem_graph, "triplet_origin", {}) or {}
    triplet2code = getattr(mem_graph, "triplet2code", {}) or {}
    mem_card_by_id = {c["id"]: c for c in cards_data.get("cards", [])}

    def _idx_from_typed_id(tid: str) -> int | None:
        try:
            return int(tid.rsplit("_", 1)[-1])
        except Exception:
            return None

    selection = sel_data.get("selection", [])
    per_paper_counts: dict[str, int] = {}
    retrieved_rows: list[dict] = []
    seen_keys: set[str] = set()
    n_with_code = 0
    for row in selection:
        mem_id = row.get("memory_card_id")
        card = mem_card_by_id.get(mem_id, {})
        origins: list[str] = []
        for tid in card.get("member_triplet_ids", []):
            i = _idx_from_typed_id(tid)
            if i is None or i < 0 or i >= len(mem_graph.triplets):
                continue
            t = mem_graph.triplets[i]
            key = mem_graph.str(t)
            origin = origin_map.get(key)
            if origin and origin not in origins:
                origins.append(origin)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            link = triplet2code.get(key)
            if link:
                n_with_code += 1
            retrieved_rows.append({
                "memory_card_id": mem_id,
                "paper_card_id": row.get("paper_card_id"),
                "score": row.get("score"),
                "triplet": [t[0], t[1], t[2]],
                "triplet_str": key,
                "origin_paper": origin,
                "linked_code": link,
            })
        row["origin_papers"] = origins
        for o in origins:
            per_paper_counts[o] = per_paper_counts.get(o, 0) + 1

    dump_path = os.path.join(paper_out, "retrieved_triplets_with_origin_and_code.json")
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump({
                "current_paper": current_slug,
                "total_retrieved_triplets": len(retrieved_rows),
                "retrieved_triplets_with_code": n_with_code,
                "rows": retrieved_rows,
            }, f, indent=2)
        log(
            f"Saved retrieved triplets+origin+code dump to {dump_path} "
            f"({len(retrieved_rows)} triplets, {n_with_code} with linked code)"
        )
    except Exception as e:
        log(f"Retrieval origins: failed to write {dump_path} ({e})")

    try:
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(sel_data, f, indent=2)
    except Exception as e:
        log(f"Retrieval origins: failed to write annotated {sel_path} ({e})")

    if not selection:
        log(f"Retrieval origins for {current_slug!r}: 0 selected memory cards")
        return

    breakdown = ", ".join(f"{p}={n}" for p, n in sorted(per_paper_counts.items(), key=lambda x: -x[1]))
    log(
        f"Retrieval origins for {current_slug!r}: {len(selection)} selected mem cards; "
        f"origin paper counts: {breakdown or '(none resolved)'}"
    )
    for row in selection[:20]:
        log(
            f"  mem_card={row.get('memory_card_id')} <- paper_card={row.get('paper_card_id')} "
            f"score={row.get('score'):.3f} from={row.get('origin_papers') or ['?']}"
        )


def _load_paper_graph(slug: str, paper_graph_json: str, model: str, api_key: str, base_url: str, log, device: str, emb_cache: bool = True, llm_timeout_s: float = 300.0):
    """Build an empty paper-type graph and load triplets from disk."""
    paper_graph = tpr.ReproductionGraph(
        model,
        "You are a helpful assistant specializing in research reproduction",
        api_key,
        log,
        base_url,
        device,
        type="paper",
    )
    paper_graph.emb_cache = emb_cache
    paper_graph.client = paper_graph.client.with_options(timeout=llm_timeout_s)
    if not os.path.isfile(paper_graph_json):
        raise FileNotFoundError(f"Missing paper graph for {slug}: {paper_graph_json}")
    paper_graph.load_triplets_from_json(paper_graph_json, clear_first=True)
    log(f"Loaded paper graph for {slug}: {len(paper_graph.triplets)} triplets from {paper_graph_json}")
    return paper_graph


def run_continual(
    *,
    bootstrap_mem_json: str,
    paper_slugs: list[str],
    paper_graph_base: str,
    paperbench_root: str,
    output_root: str,
    device: str,
    skip_linking: bool,
    skip_code_to_triplets: bool,
    max_chunk_chars: int,
    max_chunks_total: int,
    resume: bool,
    log: Logger,
    emb_cache: bool = True,
    llm_timeout_s: float = 300.0,
) -> None:
    if len(paper_slugs) < 1:
        raise ValueError("Need at least one paper slug (the bootstrap / first paper).")

    base_url = os.getenv("OPENAI_API_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")

    os.makedirs(output_root, exist_ok=True)
    resume_seed = None
    mem_graph = tpr.ReproductionGraph(
        model,
        "You are a helpful assistant specializing in research replication",
        api_key,
        log,
        base_url,
        device,
        type="mem",
    )
    mem_graph.emb_cache = emb_cache
    # Bound per-call wall-clock so a stalled/looping LLM call raises instead of blocking for many minutes.
    mem_graph.client = mem_graph.client.with_options(timeout=llm_timeout_s)
    log(f"LLM per-call timeout: {llm_timeout_s}s (on timeout the failing unit is skipped)")
    # mem_graph.load_triplets_from_json(os.path.abspath(bootstrap_mem_json), clear_first=True)
    # log(f"Bootstrap theory graph from {bootstrap_mem_json} ({len(mem_graph.triplets)} triplets, {len(mem_graph.triplet2code)} code links)")

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

    # Provenance map: triplet str-key -> originating paper slug.
    mem_graph.triplet_origin = {}
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
            _seed_data = json.load(f)
        saved_origin = _seed_data.get("triplet_origin") or {}
    except Exception as e:
        log(f"triplet_origin: failed to read sidecar from seed JSON ({e}); will tag all seed triplets with bootstrap slug")
        saved_origin = {}
    bootstrap_slug = paper_slugs[0]
    for t in mem_graph.triplets:
        key = mem_graph.str(t)
        mem_graph.triplet_origin[key] = saved_origin.get(key, bootstrap_slug)
    log(
        f"triplet_origin: {len(mem_graph.triplet_origin)} entries "
        f"({sum(1 for v in mem_graph.triplet_origin.values() if v == bootstrap_slug)} tagged as bootstrap {bootstrap_slug!r})"
    )
    if completed_slugs:
        log(f"Resume: skipping {len(completed_slugs)} already-completed paper(s): {sorted(completed_slugs)}")




    bootstrap_name = Path(bootstrap_mem_json).resolve().parent.name
    if paper_slugs[0] != bootstrap_name:
        log(
            f"Note: first slug in --papers ({paper_slugs[0]!r}) differs from bootstrap parent dir name ({bootstrap_name!r}); "
            "continuing anyway (bootstrap JSON is authoritative)."
        )

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
        #paper_md = os.path.join(paperbench_root, slug, "paper.md")
        paper_out = os.path.join(output_root, slug)
        os.makedirs(paper_out, exist_ok=True)

        paper_graph = _load_paper_graph(slug, paper_graph_path, model, api_key, base_url, log, device, emb_cache=emb_cache, llm_timeout_s=llm_timeout_s)
 
        log("Phase A: retrieval + code generation")
        tpr.generate_code_from_graph_with_retrieval(
            mem_graph=mem_graph,
            paper_graph=paper_graph,
            log=log,
            output_dir=paper_out,
            graph_base_dir=paper_out,
            retrieval_topk=3,
        )
        _log_retrieval_origins(mem_graph, paper_out, slug, log)

        repo_root = os.path.join(paper_out, "submission")
        new_code_obs_keys: set[str] | None = None
        if os.path.isdir(repo_root):
            code_files = reprutil.collect_code_files(repo_root)
            if not code_files:
                log("Phase B: skipped (no code files under submission/)")
            else:
                if skip_code_to_triplets:
                    log("Phase B1: skipped (--skip-code-to-triplets)")
                    new_code_obs_keys = None
                else:
                    log(f"Phase B1: theory triplets from new code ({len(code_files)} files, strict dedup)")
                    new_code_obs_keys = append_theory_triplets_from_generated_code_strict(
                        mem_graph,
                        repo_root,
                        log,
                        current_slug=slug,
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
                    )
        else:
            log("Phase B: skipped (submission directory missing)")

        cumulative_path = os.path.join(output_root, f"mem_graph_after_{slug}.json")
        _save_mem_graph(mem_graph, cumulative_path, slug, log)
        _save_mem_graph(mem_graph, os.path.join(output_root, "mem_graph_latest.json"), slug, log)

        log(f"Paper {slug} done in {time() - t0:.1f}s; outputs under {paper_out}")

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
        "--paperbench-root",
        type=str,
        default="/home/asagirova/arigraph/frontier-evals/project/paperbench/data/papers",
        help="Root with {slug}/paper.md (for linking pass metadata; optional if paper.md exists).",
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
        "--emb-cache",
        dest="emb_cache",
        type=tpr.str2bool,
        default=True,
        help="Cache triplet embeddings to _emb.npz sidecars to skip re-embedding on load (default: true). Disable with --emb-cache false.",
    )
    parser.add_argument(
        "--llm-timeout-s",
        dest="llm_timeout_s",
        type=float,
        default=300.0,
        help="Per-call LLM wall-clock ceiling in seconds (default: 300). On timeout the failing unit is skipped "
        "(B1 extraction chunk / B2 linking chunk) and the run proceeds. Loosen with --llm-timeout-s 600.",
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
            paperbench_root=os.path.abspath(args.paperbench_root),
            output_root=out,
            device=args.device,
            skip_linking=args.skip_linking,
            skip_code_to_triplets=args.skip_code_to_triplets,
            max_chunk_chars=args.max_code_chunk_chars,
            max_chunks_total=args.max_code_chunks_total,
            resume=args.resume,
            log=log,
            emb_cache=args.emb_cache,
            llm_timeout_s=args.llm_timeout_s,
        )
    except Exception as e:
        log(f"Continual run failed: {e}")
        raise


if __name__ == "__main__":
    main()