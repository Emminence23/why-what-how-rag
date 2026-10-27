"""Prompt templates (from the report's appendix, with a domain line per benchmark)."""
from .benchmarks import DOMAINS, LETTERS


def _block(item):
    parts = []
    if item.context:
        parts.append(f"Context:\n{item.context.strip()}\n")
    parts.append(f"Question:\n{item.question.strip()}\n")
    parts.append("Options:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item.options)))
    return "\n".join(parts)


def decompose(item, bench):
    return (
        "You decompose multiple-choice questions into focused web-search queries.\n"
        f"Topic: {DOMAINS[bench]}.\n\n"
        f"{_block(item)}\n\n"
        "Rewrite the question as three short queries that a search engine could answer:\n"
        "- WHY: causes, reasons, or motivations.\n"
        "- WHAT: key definitions, facts, or entities.\n"
        "- HOW: mechanisms, processes, or procedures.\n\n"
        "Reply with exactly three lines, starting with 'WHY:', 'WHAT:' and 'HOW:'.\n"
        "Do NOT answer the question. Do NOT add extra text."
    )


def hints(item, bench, queries, snippets):
    q = "\n".join(f"{k}: {v}" for k, v in queries.items())
    s = "\n".join(f"[{i}] ({x['tag']}) {x['snippet']}" for i, x in enumerate(snippets, 1))
    return (
        "You summarize web snippets into short, verifiable hints for a larger QA model.\n"
        f"Topic: {DOMAINS[bench]}.\n\n"
        f"{_block(item)}\n\n"
        f"Sub-queries:\n{q}\n\n"
        f"Retrieved snippets:\n{s}\n\n"
        "Using ONLY the information in the snippets, write 4-8 concise fact-style bullet points "
        "that would help answer the question. Each bullet must start with '- ', contain one fact, "
        "and avoid speculation. Do NOT answer the question. Return ONLY the bullet points."
    )


def hints_closed_book(item, bench, queries):
    """Same job as hints(), but with no web snippets: the SLM uses only what it knows."""
    q = "\n".join(f"{k}: {v}" for k, v in queries.items())
    return (
        "You write short, verifiable hints for a larger QA model.\n"
        f"Topic: {DOMAINS[bench]}.\n\n"
        f"{_block(item)}\n\n"
        f"Sub-queries:\n{q}\n\n"
        "From your own knowledge, write 4-8 concise fact-style bullet points that would help "
        "answer the question. Each bullet must start with '- ', contain one fact, and avoid "
        "speculation. Do NOT answer the question. Return ONLY the bullet points."
    )


def answer(item, bench, mode, hint_list=None):
    """Final-answer prompt for the large model. mode: 'cot' or 'instruction'."""
    letters = ", ".join(LETTERS[: len(item.options)])
    text = (
        "You are an expert at answering multiple-choice questions.\n"
        f"Topic: {DOMAINS[bench]}.\n\n{_block(item)}\n"
    )
    if hint_list:
        text += "\nHints (you may use or ignore these):\n" + "\n".join(f"- {h}" for h in hint_list) + "\n"
    if mode == "cot":
        text += ("\nThink step by step, then finish with a last line of the form\n"
                 f"Final Answer: X\nwhere X is one of {letters}.")
    elif mode == "instruction":
        text += f"\nChoose the single best answer. Reply with ONLY one letter from {letters}."
    else:
        raise ValueError(mode)
    return text
