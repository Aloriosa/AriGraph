#!/usr/bin/env python3
"""
Compare code-generation token budgets between:

- continual setting: one long run log produced by `continual_paper_reproduction.py`
- from-scratch setting: per-paper logs produced by `test_paper_reproduction.py`

We extract *only* the "Code generation tokens: <completion> completion, <prompt> prompt" lines
from log files (these correspond to the Phase A / Phase 3 code generation call).

Outputs a CSV with one row per paper slug.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_RE_CONTINUAL_STEP = re.compile(r"CONTINUAL STEP\s+(\d+)/(\d+):\s+paper\s+'([^']+)'")
_RE_CODEGEN_TOKENS = re.compile(
    r"Code generation tokens:\s*(\d+)\s*completion,\s*(\d+)\s*prompt", re.IGNORECASE
)


@dataclass(frozen=True)
class TokenStats:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_continual_log(path: str) -> dict[str, TokenStats]:
    """
    Parse a continual run log into {paper_slug -> TokenStats} using:
      - "CONTINUAL STEP ... paper 'slug'"
      - next "Code generation tokens: ..." line
    """
    out: dict[str, TokenStats] = {}
    current_slug: str | None = None

    for line in _read_text(path).splitlines():
        m_step = _RE_CONTINUAL_STEP.search(line)
        if m_step:
            # Regex groups: (step_idx, total_steps, slug)
            current_slug = m_step.group(3).strip()
            continue

        m_tok = _RE_CODEGEN_TOKENS.search(line)
        if m_tok and current_slug:
            completion = int(m_tok.group(1))
            prompt = int(m_tok.group(2))
            # Only record the first codegen token line per slug within the continual log.
            out.setdefault(current_slug, TokenStats(prompt_tokens=prompt, completion_tokens=completion))
            continue

    return out


def parse_continual_order(path: str) -> dict[str, int]:
    """
    Parse a continual run log into {paper_slug -> order_index} where order_index is the
    integer from "CONTINUAL STEP <i>/<N>: paper 'slug'".
    """
    order: dict[str, int] = {}
    for line in _read_text(path).splitlines():
        m = _RE_CONTINUAL_STEP.search(line)
        if not m:
            continue
        step_idx = int(m.group(1))
        slug = m.group(3).strip()
        # Keep first seen order if duplicates appear.
        order.setdefault(slug, step_idx)
    return order


def parse_scratch_log(path: str) -> TokenStats | None:
    """
    Parse a per-paper log and return the *last* "Code generation tokens: ..." entry
    (some logs can contain multiple runs/experiments).
    """
    last: TokenStats | None = None
    for line in _read_text(path).splitlines():
        m = _RE_CODEGEN_TOKENS.search(line)
        if not m:
            continue
        completion = int(m.group(1))
        prompt = int(m.group(2))
        last = TokenStats(prompt_tokens=prompt, completion_tokens=completion)
    return last


def iter_scratch_logs(scratch_root: str) -> Iterable[tuple[str, str]]:
    """
    Yield (slug, log_path) for {scratch_root}/{slug}/log.txt.
    """
    root = Path(scratch_root)
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        log_path = child / "log.txt"
        if log_path.is_file():
            yield (child.name, str(log_path))


def write_csv(
    out_csv: str,
    slugs: list[str],
    continual: dict[str, TokenStats],
    scratch: dict[str, TokenStats],
    continual_order: dict[str, int],
    *,
    continual_label: str,
    scratch_label: str,
) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper",
                "continual_order",
                f"{scratch_label}_prompt_tokens",
                f"{scratch_label}_completion_tokens",
                f"{scratch_label}_total_tokens",
                f"{continual_label}_prompt_tokens",
                f"{continual_label}_completion_tokens",
                f"{continual_label}_total_tokens",
                "continual_minus_scratch_total_tokens",
                "continual_div_scratch_total_tokens",
            ],
        )
        w.writeheader()

        for slug in slugs:
            s = scratch.get(slug)
            c = continual.get(slug)
            s_total = s.total_tokens if s else None
            c_total = c.total_tokens if c else None

            diff = None
            ratio = None
            if s_total is not None and c_total is not None and s_total != 0:
                diff = c_total - s_total
                ratio = c_total / s_total

            w.writerow(
                {
                    "paper": slug,
                    "continual_order": (continual_order.get(slug, "")),
                    f"{scratch_label}_prompt_tokens": (s.prompt_tokens if s else ""),
                    f"{scratch_label}_completion_tokens": (s.completion_tokens if s else ""),
                    f"{scratch_label}_total_tokens": (s_total if s_total is not None else ""),
                    f"{continual_label}_prompt_tokens": (c.prompt_tokens if c else ""),
                    f"{continual_label}_completion_tokens": (c.completion_tokens if c else ""),
                    f"{continual_label}_total_tokens": (c_total if c_total is not None else ""),
                    "continual_minus_scratch_total_tokens": (diff if diff is not None else ""),
                    "continual_div_scratch_total_tokens": (f"{ratio:.4f}" if ratio is not None else ""),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare continual vs from-scratch codegen token budgets from logs.")
    ap.add_argument(
        "--continual-log",
        required=True,
        help="Path to continual run log.txt (e.g. arigraph/continual_runs/<run>/log.txt).",
    )
    ap.add_argument(
        "--scratch-root",
        required=True,
        help="Directory containing per-paper folders with log.txt (e.g. arigraph/rewritten-.../).",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (must be under /home/asagirova/...).",
    )
    ap.add_argument(
        "--papers",
        nargs="*",
        default=None,
        help="Optional list of paper slugs. If omitted, inferred as union of slugs seen in logs.",
    )
    ap.add_argument("--continual-label", default="continual", help="CSV column prefix for continual.")
    ap.add_argument("--scratch-label", default="from_scratch", help="CSV column prefix for from-scratch.")
    args = ap.parse_args()

    continual = parse_continual_log(args.continual_log)
    continual_order = parse_continual_order(args.continual_log)

    scratch: dict[str, TokenStats] = {}
    for slug, log_path in iter_scratch_logs(args.scratch_root):
        stats = parse_scratch_log(log_path)
        if stats is not None:
            scratch[slug] = stats

    if args.papers:
        slugs = list(args.papers)
    else:
        slugs = sorted(set(continual.keys()) | set(scratch.keys()))

    # Basic sanity: output location inside /home/asagirova (workspace rule).
    out_csv = os.path.abspath(args.out_csv)
    if not out_csv.startswith("/home/asagirova/"):
        raise SystemExit(f"--out-csv must be under /home/asagirova/, got: {out_csv}")

    write_csv(
        out_csv,
        slugs,
        continual=continual,
        scratch=scratch,
        continual_order=continual_order,
        continual_label=args.continual_label,
        scratch_label=args.scratch_label,
    )

    missing_c = [s for s in slugs if s not in continual]
    missing_s = [s for s in slugs if s not in scratch]
    print(f"Wrote {out_csv} with {len(slugs)} papers.")
    if missing_c:
        print(f"Warning: missing continual token lines for {len(missing_c)} paper(s): {', '.join(missing_c[:20])}")
    if missing_s:
        print(f"Warning: missing from-scratch token lines for {len(missing_s)} paper(s): {', '.join(missing_s[:20])}")


if __name__ == "__main__":
    main()

