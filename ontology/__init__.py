"""Ontology module for cookbook graph structure and validation."""

from .cookbook_ontology import (
    ALLOWED_RELATIONS,
    PATTERN_RELATIONS,
    MAPPING_RELATIONS,
    ENTITY_RESOLUTION_RELATIONS,
    validate_triplet,
    normalize_relation,
)

__all__ = [
    "ALLOWED_RELATIONS",
    "PATTERN_RELATIONS",
    "MAPPING_RELATIONS",
    "ENTITY_RESOLUTION_RELATIONS",
    "validate_triplet",
    "normalize_relation",
]
