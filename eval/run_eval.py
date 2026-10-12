"""Measure quality. This produces the numbers for your README.

  python eval/run_eval.py --retrieval-only    # fast: compares bm25 vs vector vs hybrid (needs embeddings only)
  python eval/run_eval.py                      # full pipeline: answers, citations, refusals

Stop the API first (local Qdrant allows one process at a time).
Edit data/eval.jsonl: one JSON per line. expected_source is a KB filename, "ticket:T-1001",
or null when the question is NOT answerable and the bot should escalate.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from agents import SupportCopilot  # noqa: E402
from retriever import HybridRetriever  # noqa: E402


def load_rows():
    lines = (ROOT / "data" / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def retrieval_report(retriever, rows, k):
    answerable = [r for r in rows if r["expected_source"]]
    print(f"\nRetrieval, {len(answerable)} answerable questions")
    for mode in ("bm25", "vector", "hybrid"):
        ok = sum(r["expected_source"] in [h["source"] for h in
                 retriever.search(r["question"], r.get("role", "customer"), k=k, mode=mode)] for r in answerable)
        print(f"  {mode:<7} Hit@{k}: {ok}/{len(answerable)} = {ok / max(len(answerable), 1):.0%}")

    def top_score(r):
        hits = retriever.search(r["question"], r.get("role", "customer"), k=k)
        return max((h["vector_score"] for h in hits), default=0.0)

    yes = [top_score(r) for r in answerable]
    no = [top_score(r) for r in rows if not r["expected_source"]]
    print("\nTop vector score (use this to pick MIN_VECTOR_SCORE)")
    if yes:
        print(f"  answerable   : min {min(yes):.2f}  mean {statistics.mean(yes):.2f}")
    if no:
        print(f"  unanswerable : max {max(no):.2f}  mean {statistics.mean(no):.2f}")
    print(f"  current MIN_VECTOR_SCORE = {cfg.MIN_VECTOR_SCORE}")


def end_to_end(copilot, rows):
    cfg.ESCALATIONS_FILE = cfg.STORAGE / "eval_escalations.jsonl"   # keep eval out of real escalations
    n_ans = n_no = answered = cited_ok = refused_ok = 0
    for r in rows:
        res = copilot.ask(r["question"], r.get("role", "customer"))
        if r["expected_source"]:
            n_ans += 1
            if res["status"] == "answered":
                answered += 1
                cited_ok += r["expected_source"] in [c["source"] for c in res["citations"]]
        else:
            n_no += 1
            refused_ok += res["status"] == "escalated"
        print(f"  [{res['status']:<9}] {r['question']}")
    print("\nEnd to end")
    print(f"  answered when answerable   : {answered}/{n_ans}")
    print(f"  cited the right source     : {cited_ok}/{n_ans}")
    print(f"  refused when unanswerable  : {refused_ok}/{n_no}   <- hallucination guard")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("-k", type=int, default=cfg.TOP_K)
    args = p.parse_args()
    rows = load_rows()
    retriever = HybridRetriever()
    retrieval_report(retriever, rows, args.k)
    if not args.retrieval_only:
        end_to_end(SupportCopilot(retriever), rows)
    retriever.client.close()
