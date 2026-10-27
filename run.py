"""Answer the same questions with each arm and save every answer.

  python run.py --bench hellaswag --mode instruction --n 100
  python run.py --bench musr --arms base,nosearch,search

Arms: base (large model alone), nosearch (SLM hints, no web), search (SLM hints
from web snippets). Writes results/<bench>_<mode>.jsonl, one line per question.
Re-running skips questions already saved, so a run cut short by rate limits can
be resumed (keep the same --arms when resuming).
Needs GEMINI_API_KEY, SERPER_API_KEY for the search arm, and HF_TOKEN for Llama / GPQA.
"""
import argparse
import json
from pathlib import Path

from wwh_rag.benchmarks import DOMAINS, LETTERS, load_benchmark
from wwh_rag.pipeline import ARMS, run_arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--mode", default="cot", choices=["cot", "instruction"])
    ap.add_argument("--arms", default="base,search",
                    help=f"comma-separated, from {','.join(ARMS)} (default: base,search)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slm", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--llm", default="gemini-2.5-flash-lite")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not arms or any(x not in ARMS for x in arms):
        ap.error(f"--arms must be a comma-separated subset of {ARMS}")

    out = Path(a.out) / f"{a.bench}_{a.mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open()}

    items = [it for it in load_benchmark(a.bench, a.n, a.seed) if it.id not in done]
    print(f"{a.bench}/{a.mode} arms={arms}: {len(done)} done, {len(items)} to go")
    if not items:
        return

    from wwh_rag.models import Gemini, LocalSLM, SerperSearch
    llm = Gemini(a.llm)
    slm = LocalSLM(a.slm) if set(arms) & {"nosearch", "search"} else None
    search = SerperSearch() if "search" in arms else None

    with out.open("a") as f:
        for i, it in enumerate(items, 1):
            res = run_arms(it, a.bench, a.mode, arms, llm, slm, search)
            row = {"id": it.id, "bench": a.bench, "mode": a.mode, "slm": a.slm, "llm": a.llm,
                   "gold": it.gold_letter, "n_options": len(it.options), "arms": {}}
            for arm, r in res.items():
                r["pred"] = None if r["pred"] is None else LETTERS[r["pred"]]
                row["arms"][arm] = r
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i}/{len(items)}] gold={row['gold']} "
                  + " ".join(f"{arm}={r['pred']}" for arm, r in row["arms"].items()))
    print("Gemini tokens:", llm.tokens)


if __name__ == "__main__":
    main()
