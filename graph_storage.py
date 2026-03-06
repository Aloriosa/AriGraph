"""
Graph storage and merge logic for the cookbook knowledge graph.
Handles persistence, entity resolution (compatibility check), and merge/deduplication.
"""

import json
import os
import re
import hashlib
from pathlib import Path

from utils.utils import clear_triplet


# Default cookbook path (under home, per workspace rules)
DEFAULT_COOKBOOK_PATH = os.path.join(os.path.expanduser("~"), "arigraph", "reproduction_cookbook", "cookbook_graph.json")


def get_cookbook_path(config=None):
    """Resolve cookbook path from config, args, or default."""
    if config and hasattr(config, "cookbook_path") and config.cookbook_path:
        return config.cookbook_path
    return DEFAULT_COOKBOOK_PATH


def _triplet_to_tuple(triplet):
    """Convert triplet to hashable tuple for deduplication."""
    subj, obj, rel_data = triplet
    rel = rel_data.get("label", "") if isinstance(rel_data, dict) else str(rel_data)
    return (str(subj).strip(), str(obj).strip(), rel.strip().lower())


def _triplet_from_dict(d):
    """Reconstruct triplet from JSON dict: [subj, obj, {label: rel}]."""
    subj = d.get("subj", d.get(0, ""))
    obj = d.get("obj", d.get(1, ""))
    rel = d.get("rel", d.get("label", d.get(2, "")))
    if isinstance(rel, dict):
        rel = rel.get("label", "")
    return [subj, obj, {"label": rel}]


def load_cookbook_graph(path):
    """
    Load cookbook from JSON.
    Returns (triplets, hypernode_store, metadata).
    If file missing, returns ([], {}, {}).
    """
    if not path or not os.path.exists(path):
        return [], {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return [], {}, {}
    triplets = []
    for t in data.get("triplets", []):
        if isinstance(t, list) and len(t) >= 3:
            triplets.append([t[0], t[1], t[2] if isinstance(t[2], dict) else {"label": str(t[2])}])
        elif isinstance(t, dict):
            triplets.append(_triplet_from_dict(t))
    hypernode_store = data.get("hypernode_store", {})
    metadata = data.get("metadata", {})
    return triplets, hypernode_store, metadata


def save_cookbook_graph(path, triplets, hypernode_store, metadata=None):
    """Write cookbook to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "triplets": [[t[0], t[1], t[2]] for t in triplets],
        "hypernode_store": hypernode_store,
        "metadata": metadata or {},
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _format_triplets_for_prompt(triplets):
    """Format triplets as bullet list for LLM prompt."""
    lines = []
    for t in triplets:
        subj, obj, rel_data = t
        rel = rel_data.get("label", "") if isinstance(rel_data, dict) else str(rel_data)
        lines.append(f"- {subj}, {rel}, {obj}")
    return "\n".join(lines) if lines else "(none)"


def check_compatibility(existing_triplets, new_triplets, pattern_name, graph, log):
    """
    LLM-based compatibility check: are existing and new implementations of the same pattern compatible?
    Returns True if compatible (keep existing, skip new), False if incompatible (create variant).
    """
    from prompts.cookbook_extraction_prompt import prompt_pattern_compatibility_check

    existing_str = _format_triplets_for_prompt(existing_triplets)
    new_str = _format_triplets_for_prompt(new_triplets)
    prompt = prompt_pattern_compatibility_check.format(
        pattern_name=pattern_name,
        existing_triplets=existing_str,
        new_triplets=new_str,
    )
    try:
        response, _ = graph.generate(prompt, t=0)
        first_line = (response.strip().split("\n")[0] or "").strip().upper()
        if "COMPATIBLE" in first_line:
            log(f"  Compatibility check: {pattern_name} -> COMPATIBLE (keeping existing)")
            return True
        if "INCOMPATIBLE" in first_line:
            reason = response.strip().split("\n")[1] if "\n" in response.strip() else ""
            log(f"  Compatibility check: {pattern_name} -> INCOMPATIBLE ({reason})")
            return False
        # Default: treat unknown as incompatible to be safe
        log(f"  Compatibility check: {pattern_name} -> unclear response, treating as INCOMPATIBLE")
        return False
    except Exception as e:
        log(f"  Compatibility check error for {pattern_name}: {e}")
        return False


def _triplets_for_pattern(triplets, pattern_name):
    """Get all triplets where pattern_name appears as subject or object."""
    pattern_lower = pattern_name.lower().strip()
    result = []
    for t in triplets:
        subj = str(t[0]).lower().strip()
        obj = str(t[1]).lower().strip()
        if subj == pattern_lower or obj == pattern_lower:
            result.append(t)
    return result


def _extract_pattern_names_from_triplets(triplets):
    """Extract pattern-like subjects from triplets (for entity resolution)."""
    patterns = set()
    for t in triplets:
        subj = str(t[0]).strip()
        if subj and subj not in ("none", "unknown"):
            patterns.add(subj)
    return patterns


def merge_triplets_into_cookbook(
    existing_triplets,
    new_triplets,
    existing_hypernodes,
    new_hypernodes,
    graph,
    log,
    ontology=None,
    run_compatibility_check=True,
):
    """
    Merge new triplets into existing cookbook with entity resolution.
    Returns (merged_triplets, merged_hypernodes).
    """
    from ontology.cookbook_ontology import validate_triplet

    merged_triplets = list(existing_triplets)
    merged_hypernodes = dict(existing_hypernodes)
    seen = {_triplet_to_tuple(t) for t in existing_triplets}

    # Normalize and deduplicate new triplets
    new_normalized = []
    for t in new_triplets:
        t_cleared = clear_triplet(t)
        if ontology and not validate_triplet(t_cleared):
            log(f"  Skipping invalid triplet: {t_cleared}")
            continue
        key = _triplet_to_tuple(t_cleared)
        if key in seen:
            continue
        new_normalized.append(t_cleared)

    # Group new triplets by pattern (subject)
    patterns_in_new = {}
    for t in new_normalized:
        subj = str(t[0]).strip().lower()
        if subj not in patterns_in_new:
            patterns_in_new[subj] = []
        patterns_in_new[subj].append(t)

    # Entity resolution: for each pattern in new triplets
    resolved_skip = set()  # patterns we skip (compatible)
    resolved_variant = {}   # pattern -> variant_name for incompatible

    for pattern_name, triplets_for_pattern in patterns_in_new.items():
        existing_for_pattern = _triplets_for_pattern(merged_triplets, pattern_name)

        if existing_for_pattern and run_compatibility_check:
            compatible = check_compatibility(
                existing_for_pattern, triplets_for_pattern, pattern_name, graph, log
            )
            if compatible:
                resolved_skip.add(pattern_name)
                for t in triplets_for_pattern:
                    seen.add(_triplet_to_tuple(t))
                continue

            # Incompatible: create variant
            variant_name = f"{pattern_name}_variant"
            suffix = 1
            while any(
                str(x[0]).lower() == variant_name.lower() or str(x[1]).lower() == variant_name.lower()
                for x in merged_triplets
            ):
                variant_name = f"{pattern_name}_variant{suffix}"
                suffix += 1
            resolved_variant[pattern_name] = variant_name

    # Add triplets: skip resolved_skip, rewrite resolved_variant, add rest
    for t in new_normalized:
        subj = str(t[0]).strip()
        subj_lower = subj.lower()
        key = _triplet_to_tuple(t)
        if key in seen:
            continue
        if subj_lower in resolved_skip:
            continue
        if subj_lower in resolved_variant:
            variant_name = resolved_variant[subj_lower]
            t_rewritten = [variant_name, t[1], t[2]]
            key_rewritten = _triplet_to_tuple(t_rewritten)
            if key_rewritten not in seen:
                merged_triplets.append(t_rewritten)
                seen.add(key_rewritten)
            continue
        merged_triplets.append(t)
        seen.add(key)

    # Add variant_of triplets for incompatible patterns
    for pattern_name, variant_name in resolved_variant.items():
        variant_triplet = [variant_name, pattern_name, {"label": "variant_of"}]
        key = _triplet_to_tuple(variant_triplet)
        if key not in seen:
            merged_triplets.append(variant_triplet)
            seen.add(key)

    # Merge hypernodes
    for hid, hnode in new_hypernodes.items():
        if hid not in merged_hypernodes:
            merged_hypernodes[hid] = hnode

    return merged_triplets, merged_hypernodes


def generate_hypernode_id(pattern_name, code_snippet=""):
    """Generate unique hypernode ID."""
    h = hashlib.sha256(f"{pattern_name}:{code_snippet[:200]}".encode()).hexdigest()[:12]
    return f"hn_{pattern_name}_{h}".replace(" ", "_")
