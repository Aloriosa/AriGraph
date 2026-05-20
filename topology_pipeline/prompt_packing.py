from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .schemas import BaseCard, MemoryCard, SymbolAsset


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class PromptBudget:
    max_prompt_tokens: int = 110_000
    reserved_output_tokens: int = 120_000
    fixed_block_tokens: int = 12_000
    paper_cards_tokens: int = 28_000
    memory_cards_tokens: int = 36_000
    code_assets_tokens: int = 22_000


def _pack_block(lines: Sequence[str], token_budget: int) -> str:
    out: List[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        out.append(line)
        used += cost
    return "\n".join(out)


def build_generation_prompt(
    benchmark_rules: str,
    toy_example: str,
    paper_cards: Sequence[BaseCard],
    memory_cards: Sequence[MemoryCard],
    selected_map: Sequence[tuple[str, str, float]],
    symbol_assets: Sequence[SymbolAsset],
    budget: PromptBudget | None = None,
) -> Dict[str, object]:
    budget = budget or PromptBudget()
    asset_by_id = {a.id: a for a in symbol_assets}
    mem_by_id = {m.id: m for m in memory_cards}
    paper_by_id = {p.id: p for p in paper_cards}

    selected_paper_ids = []
    selected_memory_ids = []
    for pid, mid, _ in selected_map:
        if pid not in selected_paper_ids:
            selected_paper_ids.append(pid)
        if mid not in selected_memory_ids:
            selected_memory_ids.append(mid)

    paper_lines: List[str] = []
    for pid in selected_paper_ids:
        card = paper_by_id.get(pid)
        if not card:
            continue
        paper_lines.append(
            f"### {card.id} | intent={card.intent} | priority={card.priority}\n{card.generation_summary}"
        )
    paper_block = _pack_block(paper_lines, budget.paper_cards_tokens)

    memory_lines: List[str] = []
    for mid in selected_memory_ids:
        card = mem_by_id.get(mid)
        if not card:
            continue
        memory_lines.append(
            f"### {card.id} | intent={card.intent} | priority={card.priority}\n{card.generation_summary}"
        )
    memory_block = _pack_block(memory_lines, budget.memory_cards_tokens)

    used_assets = []
    for mid in selected_memory_ids:
        card = mem_by_id.get(mid)
        if not card:
            continue
        for aid in card.linked_asset_ids:
            if aid not in used_assets:
                used_assets.append(aid)
    asset_lines: List[str] = []
    for aid in used_assets:
        asset = asset_by_id.get(aid)
        if not asset:
            continue
        asset_lines.append(
            f"### {asset.symbol} ({asset.path}:{asset.line_start}-{asset.line_end})\n"
            f"Role tags: {', '.join(asset.role_tags)}\n"
            f"Summary: {asset.summary}\n"
            f"Code:\n```python\n{asset.code}\n```"
        )
    code_asset_block = _pack_block(asset_lines, budget.code_assets_tokens)

    prompt = (
        "You are a coding agent implementing a research paper.\n"
        "Use paper requirements first, then adapt reusable patterns, then code assets.\n\n"
        f"{benchmark_rules}\n\n"
        f"{toy_example}\n\n"
        "---\n## Paper implementation cards\n"
        f"{paper_block}\n\n"
        "---\n## Retrieved reusable memory cards\n"
        f"{memory_block}\n\n"
        "---\n## Linked code assets\n"
        f"{code_asset_block}\n\n"
        "---\nGenerate a complete repository in FILE/code-block format."
    )

    return {
        "prompt": prompt,
        "prompt_tokens_est": estimate_tokens(prompt),
        "paper_block": paper_block,
        "memory_block": memory_block,
        "code_asset_block": code_asset_block,
        "selected_paper_card_ids": selected_paper_ids,
        "selected_memory_card_ids": selected_memory_ids,
        "selected_asset_ids": used_assets,
        "budget": {
            "max_prompt_tokens": budget.max_prompt_tokens,
            "reserved_output_tokens": budget.reserved_output_tokens,
        },
    }
