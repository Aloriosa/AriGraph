"""
Parse the final answer from a generated rationale.
"""

import re
from typing import Optional


def parse_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a CoT rationale.
    Supports patterns like:
      - "The answer is 6."
      - "Answer: 6"
      - "The answer is (a)."
    Returns the captured string or None if not found.
    """
    # Look for patterns "Answer is X" or "The answer is X"
    patterns = [
        r"answer is\s+([^\.\n]+)",
        r"answer:\s*([^\.\n]+)",
        r"the answer is\s+([^\.\n]+)",
        r"the answer:\s*([^\.\n]+)",
        r"answer\s+([^\.\n]+)",
    ]
    text_lower = text.lower()
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return m.group(1).strip().strip(".").strip()
    # Fallback: take the last line that looks like a number or a letter
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    # remove trailing punctuation
    return last.rstrip(".").strip()