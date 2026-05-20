from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .schemas import SymbolAsset


def _extract_role_tags(path: str, symbol: str) -> List[str]:
    text = f"{path.lower()}::{symbol.lower()}"
    tags = []
    for tag in ("model", "trainer", "loss", "dataset", "eval", "config", "script"):
        if tag in text:
            tags.append(tag)
    return tags or ["implementation"]


def _extract_imports(code: str) -> List[str]:
    imports = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports.append(line)
    return imports


def _api_surface(code: str) -> str:
    try:
        tree = ast.parse(code)
    except Exception:
        return ""
    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            signatures.append(f"{node.name}({', '.join(args)})")
    return "; ".join(signatures[:3])


def _summary(path: str, symbol: str, code: str) -> str:
    action = "implements core logic"
    if "train" in path.lower() or "train" in symbol.lower():
        action = "runs training/update procedure"
    elif "eval" in path.lower() or "metric" in symbol.lower():
        action = "computes evaluation protocol/metrics"
    elif "config" in path.lower():
        action = "defines configuration settings"
    return f"{symbol} in {path} {action}."


def build_symbol_assets_from_triplet_links(
    triplet2code: Dict[str, object],
) -> Tuple[List[SymbolAsset], Dict[str, List[str]]]:
    assets: List[SymbolAsset] = []
    triplet_to_asset_ids: Dict[str, List[str]] = {}

    for idx, (triplet_str, payload) in enumerate(triplet2code.items()):
        if isinstance(payload, dict):
            path = str(payload.get("path", payload.get("code_location", ""))).strip()
            line_start = int(payload.get("line_start", 0) or 0)
            line_end = int(payload.get("line_end", line_start or 0) or 0)
            code = str(payload.get("code", ""))
            symbol = str(payload.get("code_location", path or f"symbol_{idx}"))
        else:
            path = ""
            line_start = 0
            line_end = 0
            code = str(payload)
            symbol = f"symbol_{idx}"

        asset_id = f"asset_{idx:05d}"
        role_tags = _extract_role_tags(path, symbol)
        asset = SymbolAsset(
            id=asset_id,
            symbol=symbol,
            path=path,
            line_start=line_start,
            line_end=line_end,
            role_tags=role_tags,
            imports=_extract_imports(code),
            api_surface=_api_surface(code),
            summary=_summary(path, symbol, code),
            code=code,
        )
        assets.append(asset)
        triplet_to_asset_ids.setdefault(triplet_str, []).append(asset_id)
    return assets, triplet_to_asset_ids
