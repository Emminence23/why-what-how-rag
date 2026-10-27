"""Summarise saved runs: accuracy of each arm and paired comparisons between arms.

  python analyze.py            # reads results/*.jsonl, writes results/summary.csv

Pairs reported when both arms were run: search vs base, nosearch vs base, and
search vs nosearch (what web retrieval adds on top of the SLM's own hints).
"""
import json
from pathlib import Path

import pandas as pd

from wwh_rag.stats import mcnemar_exact

PAIRS = [("base", "search"), ("base", "nosearch"), ("nosearch", "search")]


def summarise(recs):
    rows = []
    n = len(recs)
    for a, b in PAIRS:
        if not all(a in r["arms"] and b in r["arms"] for r in recs):
            continue
        xa = [r["arms"][a]["correct"] for r in recs]
        xb = [r["arms"][b]["correct"] for r in recs]
        only_a, only_b, p = mcnemar_exact(xa, xb)
        rows.append({
            "bench": recs[0]["bench"], "mode": recs[0]["mode"], "n": n,
            "comparison": f"{b} vs {a}", "acc_a": sum(xa) / n, "acc_b": sum(xb) / n,
            "delta_pts": 100 * (sum(xb) - sum(xa)) / n,
            "only_a_right": only_a, "only_b_right": only_b, "mcnemar_p": p,
            "unparsed": sum(r["arms"][a]["pred"] is None or r["arms"][b]["pred"] is None
                            for r in recs),
        })
    return rows


def main(folder="results"):
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        recs = [json.loads(line) for line in path.open()]
        if recs:
            rows += summarise(recs)
    if not rows:
        print(f"No results in {folder}/ yet. Run run.py first.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(Path(folder) / "summary.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
