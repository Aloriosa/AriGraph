"""Domain ontology for topology-friendly reproduction graphs."""

from __future__ import annotations

import re
from typing import Dict, Tuple


NODE_TYPES = frozenset(
    {
        "module",
        "algorithm_step",
        "hyperparameter",
        "schedule",
        "metric",
        "dataset",
        "artifact",
        "script",
        "failure_mode",
        "fix",
        "symbol",
        "unknown",
    }
)

RELATION_FAMILIES: Dict[str, str] = {
    "part_of": "structure",
    "has_component": "structure",
    "uses": "dependency",
    "requires": "dependency",
    "depends_on": "dependency",
    "configured_by": "configuration",
    "has_config": "configuration",
    "value": "configuration",
    "precedes": "procedure",
    "executes": "procedure",
    "trained_on": "data",
    "evaluated_by": "evaluation",
    "achieves": "evaluation",
    "produces": "artifact",
    "consumes": "artifact",
    "implemented_by": "mapping",
    "implemented_in": "mapping",
    "configured_in": "mapping",
    "defined_in": "mapping",
    "handles": "mapping",
    "fails_on": "diagnostics",
    "caused_by": "diagnostics",
    "solved_by": "diagnostics",
    "fixed_by": "diagnostics",
    "improves": "performance",
    "reduces": "performance",
    "increases": "performance",
    "maintains": "performance",
    "preserves": "performance",
}

ALLOWED_RELATIONS = frozenset(RELATION_FAMILIES.keys())
PATTERN_RELATIONS = frozenset(
    rel for rel, family in RELATION_FAMILIES.items() if family in {"procedure", "dependency", "configuration", "evaluation"}
)
MAPPING_RELATIONS = frozenset(
    rel for rel, family in RELATION_FAMILIES.items() if family == "mapping"
)
ENTITY_RESOLUTION_RELATIONS = frozenset({"part_of"})

RELATION_ALIASES = {
    "step 1": "precedes",
    "step1": "precedes",
    "step_1": "precedes",
    "step 2": "precedes",
    "step2": "precedes",
    "step_2": "precedes",
    "step 3": "precedes",
    "step3": "precedes",
    "step_3": "precedes",
    "typically uses": "uses",
    "has config": "has_config",
    "implemented in": "implemented_in",
    "implemented by": "implemented_by",
    "variant of": "part_of",
    "differs by": "configured_by",
    "alternative impl": "implemented_by",
    "increase_training_efficiency": "improves",
    "increase_inference_efficiency": "improves",
}

ENTITY_ALIASES = {
    "learning rate": "learning_rate",
    "lr": "learning_rate",
    "warmup steps": "warmup_steps",
    "batch size": "batch_size",
    "validation set": "val_split",
    "top-k": "top_k",
    "moe": "mixture_of_experts",
}

RELATION_TO_TYPE_HINTS: Dict[str, Tuple[str, str]] = {
    "configured_by": ("module", "hyperparameter"),
    "has_config": ("module", "hyperparameter"),
    "value": ("hyperparameter", "artifact"),
    "trained_on": ("module", "dataset"),
    "evaluated_by": ("module", "metric"),
    "implemented_in": ("module", "symbol"),
    "implemented_by": ("module", "symbol"),
    "defined_in": ("module", "symbol"),
    "fails_on": ("module", "failure_mode"),
    "fixed_by": ("failure_mode", "fix"),
}


def normalize_entity(entity: str) -> str:
    if not entity or not isinstance(entity, str):
        return ""
    canonical = re.sub(r"\s+", " ", entity.strip().lower())
    canonical = canonical.replace("-", "_").replace(" ", "_")
    return ENTITY_ALIASES.get(canonical, canonical)


def normalize_relation(rel: str) -> str:
    """Map LLM output relation to a canonical ontology relation."""
    if not rel or not isinstance(rel, str):
        return ""
    rel_norm = rel.strip().lower()
    rel_norm = RELATION_ALIASES.get(rel_norm, rel_norm)
    rel_norm = rel_norm.replace(" ", "_")
    return rel_norm if rel_norm in ALLOWED_RELATIONS else rel_norm


def infer_node_type(entity: str, relation: str = "", role: str = "subject") -> str:
    ent = normalize_entity(entity)
    rel = normalize_relation(relation)
    if rel in RELATION_TO_TYPE_HINTS:
        subj_hint, obj_hint = RELATION_TO_TYPE_HINTS[rel]
        return subj_hint if role == "subject" else obj_hint

    if any(tok in ent for tok in ("dataset", "corpus", "split")):
        return "dataset"
    if any(tok in ent for tok in ("loss", "metric", "accuracy", "bleu", "f1")):
        return "metric"
    if any(tok in ent for tok in ("lr", "learning_rate", "dropout", "batch_size", "epochs", "steps")):
        return "hyperparameter"
    if any(tok in ent for tok in ("script", "reproduce.sh", ".sh")):
        return "script"
    if any(tok in ent for tok in ("error", "failure", "nan", "oom")):
        return "failure_mode"
    if any(tok in ent for tok in ("fix", "clip", "stabilize")):
        return "fix"
    if any(tok in ent for tok in (".py", "::", "function", "method", "class")):
        return "symbol"
    return "module"


def relation_family(relation: str) -> str:
    rel = normalize_relation(relation)
    return RELATION_FAMILIES.get(rel, "other")


def validate_triplet(triplet) -> bool:
    """Check (subj, obj, rel) against ontology constraints."""
    if not triplet or len(triplet) != 3:
        return False
    subj, obj, rel_data = triplet
    rel = rel_data.get("label") if isinstance(rel_data, dict) else rel_data
    if not rel or not isinstance(rel, str):
        return False

    rel_norm = normalize_relation(rel)
    if rel_norm not in ALLOWED_RELATIONS:
        return False

    subj_norm = normalize_entity(str(subj))
    obj_norm = normalize_entity(str(obj))
    if not subj_norm or not obj_norm:
        return False

    subj_type = infer_node_type(subj_norm, rel_norm, role="subject")
    obj_type = infer_node_type(obj_norm, rel_norm, role="object")
    return subj_type in NODE_TYPES and obj_type in NODE_TYPES
