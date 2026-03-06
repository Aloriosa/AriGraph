"""
Ontology for the cookbook knowledge graph.
Defines allowed relations, entity types, and validation/normalization helpers.
"""

# Pattern layer relations (pipelines, config, evaluation)
PATTERN_RELATIONS = frozenset({
    "step_1", "step_2", "step_3", "step_4", "step_5", "step_6", "step_7", "step_8", "step_9", "step_10",
    "requires",
    "typically_uses",
    "has_config",
    "evaluated_by",
})

# Mapping layer: pattern -> hypernode (code implementation)
# Also includes legacy linking relations from prompt_paper_code_linking
MAPPING_RELATIONS = frozenset({
    "implemented_in",
    "configured_in",
    "handles",
    "executes",
    "defined_in",
})

# Entity resolution: variant patterns
ENTITY_RESOLUTION_RELATIONS = frozenset({
    "variant_of",
    "differs_by",
    "alternative_impl",
})

# Implementation knowledge (paper extraction): hyperparameters, config, procedures, results
IMPLEMENTATION_RELATIONS = frozenset({
    "value", "type", "is_a", "has", "has_component", "uses", "includes",
    "trained_on", "improves", "reduces", "increases", "maintains", "preserves",
    "achieves", "causes", "combined_with", "combines", "introduces",
    "solved_by", "caused_by", "prevents", "prunes", "keeps", "speeds_up",
    "discards", "adjusts", "allows", "tunes", "removes", "shows", "have",
    "stands_for", "adds", "integrate_with", "provides", "integrated_with",
    "reaches", "pruned_by", "compared_to", "takes", "makes", "shares",
    "does_not_provide", "fine_tuned_for", "incurs", "diminish",
    "increase_training_efficiency", "increase_inference_efficiency",
})

ALLOWED_RELATIONS = (
    PATTERN_RELATIONS | MAPPING_RELATIONS | ENTITY_RESOLUTION_RELATIONS | IMPLEMENTATION_RELATIONS
)

# Aliases for LLM output normalization (common variations -> canonical)
RELATION_ALIASES = {
    "step 1": "step_1", "step1": "step_1",
    "step 2": "step_2", "step2": "step_2",
    "step 3": "step_3", "step3": "step_3",
    "step 4": "step_4", "step4": "step_4",
    "step 5": "step_5", "step5": "step_5",
    "step 6": "step_6", "step6": "step_6",
    "step 7": "step_7", "step7": "step_7",
    "step 8": "step_8", "step8": "step_8",
    "step 9": "step_9", "step9": "step_9",
    "step 10": "step_10", "step10": "step_10",
    "implemented in": "implemented_in",
    "implemented_in": "implemented_in",
    "typically uses": "typically_uses",
    "has config": "has_config",
    "evaluated by": "evaluated_by",
    "variant of": "variant_of",
    "differs by": "differs_by",
    "alternative impl": "alternative_impl",
}


def normalize_relation(rel: str) -> str:
    """
    Map LLM output to canonical relation string.
    Returns normalized relation; if not in aliases, lowercases and replaces spaces with underscores.
    """
    if not rel or not isinstance(rel, str):
        return ""
    rel = rel.strip().lower()
    canonical = RELATION_ALIASES.get(rel)
    if canonical:
        return canonical
    # Generic normalization
    normalized = rel.replace(" ", "_")
    return normalized if normalized in ALLOWED_RELATIONS else rel


def validate_triplet(triplet) -> bool:
    """
    Check (subj, obj, rel) against ontology.
    Triplet format: [subj, obj, {"label": rel}]
    Returns True if valid, False otherwise.
    """
    if not triplet or len(triplet) != 3:
        return False
    subj, obj, rel_data = triplet
    rel = rel_data.get("label") if isinstance(rel_data, dict) else rel_data
    if not rel or not isinstance(rel, str):
        return False
    rel_norm = normalize_relation(rel)
    if rel_norm not in ALLOWED_RELATIONS:
        return False
    if not subj or not obj:
        return False
    subj_str = str(subj).strip() if subj else ""
    obj_str = str(obj).strip() if obj else ""
    return len(subj_str) > 0 and len(obj_str) > 0
