"""Read model outputs. Kept separate (and dependency-free) so it can be unit-tested."""
import re

from .benchmarks import LETTERS


def parse_queries(text, fallback):
    """WHY/WHAT/HOW lines from the SLM's *generated* text; missing ones fall back to the question."""
    out = {}
    for tag in ["WHY", "WHAT", "HOW"]:
        m = re.search(rf"^\W*{tag}\W*:(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        q = m.group(1).strip(" *_\t") if m else ""
        out[tag] = q if q and not q.startswith("<") else fallback
    return out


def parse_hints(text, limit=8):
    hints = [ln.strip().lstrip("-*•").strip() for ln in text.splitlines()
             if ln.strip()[:1] in "-*•"]
    return [h for h in hints if h][:limit]


def parse_answer(text, n_options):
    """Index of the chosen option, or None if the reply can't be read.

    Uses the last 'Final Answer: X' if present; otherwise accepts a reply that is
    just a letter (e.g. 'B', '(B)', 'B.'). Never grabs a stray capital from the reasoning.
    """
    valid = LETTERS[:n_options]
    hits = re.findall(r"final\s+answer\s*[:\-]?\s*\**\(?([A-Z])\b", text or "", flags=re.IGNORECASE)
    if hits:
        letter = hits[-1].upper()
    else:
        m = re.fullmatch(r"\W*([A-Z])\W*", (text or "").strip())
        letter = m.group(1) if m else None
    return valid.index(letter) if letter and letter in valid else None
