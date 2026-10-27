"""The three ways of answering a question, all run on the same item:

  base      the large model alone
  nosearch  SLM writes WHY/WHAT/HOW queries, then hints from its own knowledge
  search    SLM writes the queries, web search, then hints from the snippets

Comparing nosearch with search isolates what web retrieval adds on top of the
SLM's hints; comparing each with base gives the overall effect.
"""
from . import prompts
from .parsing import parse_answer, parse_hints, parse_queries

ARMS = ("base", "nosearch", "search")


def baseline(item, bench, mode, llm):
    raw = llm.generate(prompts.answer(item, bench, mode))
    return {"raw": raw, "pred": parse_answer(raw, len(item.options))}


def why_what_how(item, bench, mode, slm, llm, search=None, k=2):
    """1) SLM writes WHY/WHAT/HOW queries  2) web search for each (skipped if search is None)
    3) SLM turns the snippets (or its own knowledge) into hints  4) the large model answers."""
    slm_raw = slm.generate(prompts.decompose(item, bench), max_new_tokens=128)
    queries = parse_queries(slm_raw, fallback=item.question)

    snippets = []
    if search is not None:
        for tag, q in queries.items():
            try:
                found = search.search(q, k=k)
            except Exception as e:
                print(f"[search] {tag}: {e}")
                found = []
            snippets += [{"tag": tag, "query": q, **s} for s in found]

    if search is None:
        hint_list = parse_hints(slm.generate(prompts.hints_closed_book(item, bench, queries)))
    elif snippets:
        hint_list = parse_hints(slm.generate(prompts.hints(item, bench, queries, snippets)))
    else:
        hint_list = []

    raw = llm.generate(prompts.answer(item, bench, mode, hint_list))
    return {"queries": queries, "snippets": snippets, "hints": hint_list,
            "raw": raw, "pred": parse_answer(raw, len(item.options))}


def run_arms(item, bench, mode, arms, llm, slm=None, search=None):
    """Answer one item with each requested arm. Returns {arm: result}, each with 'correct'."""
    out = {}
    for arm in arms:
        if arm == "base":
            r = baseline(item, bench, mode, llm)
        elif arm == "nosearch":
            r = why_what_how(item, bench, mode, slm, llm, search=None)
        elif arm == "search":
            r = why_what_how(item, bench, mode, slm, llm, search=search)
        else:
            raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
        r["correct"] = r["pred"] == item.gold
        out[arm] = r
    return out
