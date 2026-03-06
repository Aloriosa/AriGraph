# Plan vs Implementation Checklist

## Implemented

| Step | Plan Item | Status | Notes |
|------|-----------|--------|-------|
| 1 | Ontology module | DONE | `ontology/cookbook_ontology.py` with ALLOWED_RELATIONS, validate_triplet, normalize_relation |
| 2 | Phase 1 prompt | DONE | `prompt_cookbook_reusable_extraction`, graph_builder_instruction updated |
| 4 | Graph persistence | DONE | `graph_storage.py`: load_cookbook_graph, save_cookbook_graph, get_cookbook_path |
| 5 | Entity resolution | DONE | check_compatibility (LLM-based), prompt_pattern_compatibility_check |
| 6 | Merge/deduplication | DONE | merge_triplets_into_cookbook in graph_storage.py |
| 7 | Integration | DONE | Load at start, save at end, --cookbook-path, --no-load-cookbook, --repo-dir |
| 8 | Ontology enforcement | DONE | ReproductionGraph.add_triplets validates before adding |
| 9 | Coding agent prompt | PARTIAL | `prompt_coding_agent_with_cookbook` exists but generate_code_from_graph does NOT use it |

## Implemented (previously missing)

### 1. Phase 2: Hypernode-based linking (Step 3) — DONE

**Implementation**: `run_linking_pass_hypernodes` uses `prompt_reusable_pattern_to_code_mapping`, parses JSON blocks via `_extract_json_blocks`, creates hypernodes in `graph.hypernode_store`, adds `(pattern, implemented_in, hypernode_id)` triplets. Main flow now calls `run_linking_pass_hypernodes` instead of `run_linking_pass`.

### 2. generate_code_from_graph (Step 9) — DONE

**Implementation**: Uses `prompt_coding_agent_with_cookbook`; adds `paper_summary=None`, `hypernode_store=None`; loads paper (first 3000 chars) if paper_summary is None; `_format_cookbook_for_agent` expands hypernodes (code, docs, imports) for the agent.

### 3. Optional: metadata in save_cookbook_graph

**Plan**: metadata includes `papers_processed` list that accumulates across runs.

**Current**: metadata overwrites with `papers_processed: [current_paper]` each run. Does not merge with existing papers_processed from loaded cookbook.

### 4. Optional: resolve_pattern function

**Plan** mentions `resolve_pattern(triplet, existing_triplets, new_triplets_batch) -> action`.

**Current**: Logic is inline in `merge_triplets_into_cookbook`. No separate function. Acceptable.

## Summary

**Critical gaps**:
1. Phase 2 hypernode linking — run_linking_pass needs to use prompt_reusable_pattern_to_code_mapping, parse JSON, create hypernodes
2. generate_code_from_graph — needs to use prompt_coding_agent_with_cookbook, paper_summary, hypernode expansion

**Minor**: metadata papers_processed could accumulate across runs.
