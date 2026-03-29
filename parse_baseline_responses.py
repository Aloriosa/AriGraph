#!/usr/bin/env python3
"""
Parse generated_code_response.txt from baseline and react Gpt-oss trial folders
and save parsed files into submission_trial{i} for each paper.

Usage:
  python parse_baseline_responses.py [--root /path/to/arigraph]

Scans for:
  - paper2code_baseline_Gpt-oss*
  - paper2code_react*Gpt-oss*

For each paper folder with generated_code_response.txt, parses it and writes
to {paper_folder}/submission_trial{i}/ where i is extracted from the parent trial folder.
"""

import argparse
import re
import shutil
from pathlib import Path

from test_paper_reproduction import _parse_code_generator_output


def _extract_trial_number(folder_name: str) -> int | None:
    """Extract trial number from folder name, e.g. trial1 -> 1, trial5 -> 5."""
    m = re.search(r"_trial(\d+)(?:s\d+-\d+)?$", folder_name)
    if m:
        return int(m.group(1))
    return None


def _find_trial_folders(root: Path):
    """Yield (trial_folder_path, trial_num) for baseline and react Gpt-oss folders."""
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if "Gpt-oss" in name:
            if name.startswith("paper2code_baseline_Gpt-oss") or name.startswith("paper2code_react"):
                if name.endswith("trials1-5"):
                    continue
        
                trial_num = int(name[-1])#_extract_trial_number(name)
                yield d, trial_num

def parse_code(resp_path, paper_dir, root, trial_num) -> bool:
    """Parse response and write to submission_trial{i}. Returns True on success."""
    submission_dir = paper_dir / f"submission_trial{trial_num}"

    for existing in paper_dir.iterdir():
        if existing.is_dir() and existing.name.startswith("submission"):
            shutil.rmtree(existing)

    try:
        response_text = resp_path.read_text(encoding="utf-8", errors="replace")
        print(str(submission_dir))
        files_created, _ = _parse_code_generator_output(response_text, str(submission_dir))
        print(f"Parsed {paper_dir.name} (trial{trial_num}): {len(files_created)} files -> {submission_dir.relative_to(root)}")
        return True
    except Exception as e:
        print(f"ERROR {paper_dir.name} (trial{trial_num}): {e}")
        return False


def check_missing_generated_reproduction(root: Path) -> list[tuple[str, Path]]:
    """Find submission_trial{i} folders that do NOT contain generated_reproduction.txt."""
    missing = []
    for trial_folder, trial_num in _find_trial_folders(root):
        trial_type = "baseline" if "baseline" in trial_folder.name else "react"
        for paper_dir in trial_folder.iterdir():
            if not paper_dir.is_dir():
                continue
            sub_dir = paper_dir / f"submission_trial{trial_num}"
            if not sub_dir.exists():
                continue
            gen_file = sub_dir / "generated_reproduction.txt"
            if not gen_file.exists():
                rel = sub_dir.relative_to(root)
                missing.append((f"{trial_type} trial{trial_num}", rel))
    return missing


def main():
    parser = argparse.ArgumentParser(description="Parse generated_code_response.txt into submission_trial{i} folders")
    parser.add_argument("--root", type=str, default="/home/asagirova/arigraph/", help="Root directory")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    root = root.resolve()

    
    total_parsed = 0
    total_errors = 0

    for trial_folder, trial_num in _find_trial_folders(root):
        for paper_dir in trial_folder.iterdir():
            if not paper_dir.is_dir():
                continue
            resp_path = paper_dir / "generated_code_response.txt"
            
            parse_code(resp_path, paper_dir, root, trial_num)
            #
            #if not resp_path.exists():
            #    continue

    

if __name__ == "__main__":
    main()
