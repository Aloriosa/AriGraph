"""Export paperbench judge scores from run trial folders to a CSV."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Trial folder names look like `{paper_id}_{uuid}` (UUID is always last).
_UUID_SUFFIX = re.compile(
    r"_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def paper_name_from_trial_folder(folder_name: str) -> str:
    """Paper slug from a trial subfolder name (prefix before the trailing _<uuid>)."""
    m = _UUID_SUFFIX.search(folder_name)
    if m:
        return folder_name[: m.start()]
    return folder_name.split("_", 1)[0]


def _read_paper_and_score(grade_path: Path) -> tuple[str, float]:
    print(f"Reading grade from {grade_path}")
    with grade_path.open(encoding="utf-8") as f:
        data = json.load(f)
    result = data["paperbench_result"]
    paper_id = result.get("paper_id") or paper_name_from_trial_folder(
        result.get("run_id", grade_path.parent.name)
    )
    if not result["judge_output"]:
        return paper_id, 0.0
    score = result["judge_output"].get("score", 0.0)
    return paper_id, float(score)


def collect_scores_by_paper(
    run_dir: Path,
    *,
    max_trials: int = 5,
) -> dict[str, list[float]]:
    """
    For each trial subfolder under ``run_dir`` that contains ``grade.json``,
    read ``paperbench_result.judge_output.score``.

    Scores are grouped by paper id. Within each paper, trials are ordered by
    subfolder name (lexicographic), then truncated to ``max_trials``.
    """
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    by_paper: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        grade_path = child / "grade.json"
        if not grade_path.is_file():
            continue
        paper_id, score = _read_paper_and_score(grade_path)
        by_paper[paper_id].append((child.name, score))

    out: dict[str, list[float]] = {}
    for paper, pairs in by_paper.items():
        pairs.sort(key=lambda x: x[0])
        scores = [s for _, s in pairs[:max_trials]]
        out[paper] = scores
    return out


def export_run_scores_csv(
    run_folders: Path | Sequence[Path],
    output_csv: Path,
    *,
    max_trials: int = 5,
    include_run_group: bool | None = None,
) -> None:
    """
    Write a CSV: one row per paper, columns ``trial_1`` … ``trial_{max_trials}``.

    Parameters
    ----------
    run_folders
        One run group directory (e.g. ``.../runs/2026-04-01-..._run-group_...``)
        or a sequence of such directories.
    output_csv
        Destination path (parent directories are created if needed).
    max_trials
        Number of trial columns (default 5).
    include_run_group
        If True, first column is the run folder basename. If False, omitted.
        If None, True when multiple ``run_folders`` are passed, else False.
    """
    if isinstance(run_folders, Path):
        paths = [run_folders]
    else:
        paths = list(run_folders)

    if include_run_group is None:
        include_run_group = len(paths) > 1

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    trial_headers = [f"trial_{i}" for i in range(1, max_trials + 1)]
    fieldnames = (["run_group", "paper"] if include_run_group else ["paper"]) + trial_headers

    rows: list[dict[str, str | float | None]] = []
    for rd in paths:
        rd = Path(rd)
        by_paper = collect_scores_by_paper(rd, max_trials=max_trials)
        for paper in sorted(by_paper.keys()):
            scores = by_paper[paper]
            row: dict[str, str | float | None] = {"paper": paper}
            if include_run_group:
                row["run_group"] = rd.name
            for i in range(max_trials):
                key = f"trial_{i + 1}"
                row[key] = scores[i] if i < len(scores) else None
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import argparse

    
    export_run_scores_csv(
        [
'/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T14-57-27-GMT_run-group_direct_submission_solver'
            ],
        '/home/asagirova/arigraph/react_skip_repro_false_code_only_false.csv',
        max_trials=5,
    )
'''
'''
# golden_repo_mem_eval_skip_repro_true_code_only_true

"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-57-00-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-49-55-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-36-16-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-34-03-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-25-34-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-13-41-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-11-01-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-08-57-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-07-35-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-05-49-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-04-13-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-02-57-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-01-17-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-19T23-55-35-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-19T23-51-44-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-19T23-48-54-GMT_run-group_direct_submission_solver",





# golden_repo_mem_eval_skip_repro_false_code_only_false
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T00-59-46-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-05-58-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-17-04-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-23-29-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-28-36-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-34-21-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-44-06-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T01-57-51-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T02-03-00-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T02-06-27-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T02-16-36-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T02-35-58-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T02-41-23-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-20T03-10-43-GMT_run-group_direct_submission_solver",






            # one pass
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-03-19T04-01-36-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-03-31T23-06-24-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-03-31T23-54-15-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-03-31T23-57-51-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T00-04-10-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T00-11-38-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T00-24-11-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T00-29-11-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T01-33-52-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T01-44-25-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T01-50-31-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-07-33-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-15-36-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-21-35-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-25-35-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-54-05-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T02-58-30-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T03-09-39-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T03-51-31-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/2026-04-01T04-19-37-GMT_run-group_direct_submission_solver",
'''

'''



'''
# react
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T19-52-49-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-16-30-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-20-55-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-24-34-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-31-51-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-48-54-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T20-55-10-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T22-04-06-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T22-16-11-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T22-32-39-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T22-41-03-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T22-45-35-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T23-23-29-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T23-29-03-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-01T23-40-26-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-02T00-20-58-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-02T00-57-10-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-07T08-34-45-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-07T08-42-36-GMT_run-group_direct_submission_solver",
"/home/asagirova/arigraph/frontier-evals/project/paperbench/runs/react-baseline/2026-04-07T08-52-05-GMT_run-group_direct_submission_solver", 
'''