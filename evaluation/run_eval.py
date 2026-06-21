"""Phase 7 — judge model outputs and produce the headline metrics.

Runs LOCALLY (only needs the Gemini judge). Consumes preds_<tag>.jsonl files
written by infer.py and reports:

  * Preference Win Rate   — pairwise judge, model_b vs model_a, position-swapped
  * Recommendation Quality — per-axis 1-5 scores (actionability / specificity /
                             business value), averaged per model

    python evaluation/run_eval.py --a results/preds_base.jsonl \
                                  --b results/preds_dpo.jsonl

Writes results/eval_report.json and results/eval_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from marketing_agent.config import JUDGE_MODEL  # noqa: E402
from marketing_agent.gemini_client import Gemini  # noqa: E402
from marketing_agent.prompts import (  # noqa: E402
    JUDGE_SYSTEM, pairwise_judge_prompt, score_judge_prompt)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path: str) -> dict:
    return {r["campaign_id"]: r for r in
            (json.loads(l) for l in Path(path).open())}


def win_rate(judge: Gemini, a: dict, b: dict) -> dict:
    """Fraction of shared campaigns where B beats A (ties excluded). Position
    swapped per item to cancel order bias."""
    ids = sorted(set(a) & set(b))
    b_wins = a_wins = ties = 0
    for i, cid in enumerate(ids):
        block = a[cid]["metrics_block"]
        b_is_first = (i % 2 == 0)
        first, second = ((b[cid]["output"], a[cid]["output"]) if b_is_first
                         else (a[cid]["output"], b[cid]["output"]))
        try:
            v = judge.generate_json(
                pairwise_judge_prompt(block, first, second), system=JUDGE_SYSTEM)
        except (ValueError, RuntimeError):
            continue
        w = str(v.get("winner", "")).upper()
        if w == "TIE" or w not in ("A", "B"):
            ties += 1
            continue
        b_won = (w == "A") if b_is_first else (w == "B")
        b_wins += int(b_won)
        a_wins += int(not b_won)
    decided = b_wins + a_wins
    return {"n_compared": len(ids), "b_wins": b_wins, "a_wins": a_wins,
            "ties": ties,
            "b_win_rate": round(b_wins / decided, 3) if decided else None}


def quality(judge: Gemini, preds: dict) -> dict:
    axes = {"actionability": [], "specificity": [], "business_value": []}
    for r in preds.values():
        try:
            s = judge.generate_json(
                score_judge_prompt(r["metrics_block"], r["output"]),
                system=JUDGE_SYSTEM)
        except (ValueError, RuntimeError):
            continue
        for k in axes:
            if isinstance(s.get(k), (int, float)):
                axes[k].append(s[k])
    out = {k: round(mean(v), 2) for k, v in axes.items() if v}
    out["overall"] = round(mean(out.values()), 2) if out else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline preds jsonl (e.g. base)")
    ap.add_argument("--b", required=True, help="candidate preds jsonl (e.g. dpo)")
    args = ap.parse_args()

    judge = Gemini(JUDGE_MODEL, temperature=0.0)
    a, b = load(args.a), load(args.b)
    tag_a = Path(args.a).stem.replace("preds_", "")
    tag_b = Path(args.b).stem.replace("preds_", "")

    report = {
        "model_a": tag_a,
        "model_b": tag_b,
        "win_rate_b_over_a": win_rate(judge, a, b),
        "quality": {tag_a: quality(judge, a), tag_b: quality(judge, b)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval_report.json").write_text(json.dumps(report, indent=2))

    wr = report["win_rate_b_over_a"]
    md = [
        f"# Evaluation — {tag_b} vs {tag_a}\n",
        f"**Preference win rate ({tag_b} over {tag_a}):** "
        f"{wr['b_win_rate']} "
        f"({wr['b_wins']}-{wr['a_wins']}-{wr['ties']} W-L-T, "
        f"n={wr['n_compared']})\n",
        "\n## Recommendation quality (LLM-as-judge, 1-5)\n",
        "| Model | Actionability | Specificity | Business Value | Overall |",
        "|---|---|---|---|---|",
    ]
    for tag in (tag_a, tag_b):
        q = report["quality"][tag]
        md.append(f"| {tag} | {q.get('actionability','-')} | "
                  f"{q.get('specificity','-')} | {q.get('business_value','-')} | "
                  f"**{q.get('overall','-')}** |")
    (RESULTS / "eval_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote results/eval_report.json and results/eval_report.md")


if __name__ == "__main__":
    main()
