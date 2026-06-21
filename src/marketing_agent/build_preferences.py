"""Phase 3 — turn candidate pairs into a DPO preference dataset.

A Gemini judge picks the better of the two candidates (with A/B order randomized
per item to cancel position bias). The winner becomes `chosen`, the loser
`rejected`. Ties are dropped. Output schema is exactly what TRL's DPOTrainer
expects: {prompt, chosen, rejected}. Output: data/preferences/preferences.jsonl

    python -m marketing_agent.build_preferences
"""
from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from . import config
from .gemini_client import Gemini
from .prompts import JUDGE_SYSTEM, pairwise_judge_prompt


def _metrics_block(prompt: str) -> str:
    """Recover the metrics block from the stored recommendation prompt."""
    # prompt starts with "Campaign metrics:\n\n<block>\n\nProvide exactly 3..."
    body = prompt.split("Campaign metrics:\n\n", 1)[-1]
    return body.split("\n\nProvide exactly", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO preference pairs.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    judge = Gemini(config.JUDGE_MODEL, temperature=0.0)
    config.DATA_PREFS.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in config.RECOMMENDATIONS_JSONL.open()]
    if args.limit:
        records = records[: args.limit]

    kept = ties = 0
    with config.PREFERENCES_JSONL.open("w") as out:
        for i, r in enumerate(tqdm(records, desc="judge")):
            block = _metrics_block(r["prompt"])
            # Deterministically alternate which candidate is shown as "A" to
            # cancel position bias without needing RNG.
            strong_is_a = (i % 2 == 0)
            a, b = (r["strong"], r["weak"]) if strong_is_a else (r["weak"], r["strong"])
            try:
                verdict = judge.generate_json(
                    pairwise_judge_prompt(block, a, b), system=JUDGE_SYSTEM)
            except (ValueError, RuntimeError):
                continue  # unparseable judge output -> skip
            winner = str(verdict.get("winner", "")).upper()
            if winner not in ("A", "B"):
                ties += 1
                continue
            winner_text = a if winner == "A" else b
            loser_text = b if winner == "A" else a
            out.write(json.dumps({
                "prompt": r["prompt"],
                "chosen": winner_text,
                "rejected": loser_text,
                "campaign_id": r["campaign_id"],
                "judge_reason": verdict.get("reason", ""),
            }) + "\n")
            kept += 1

    print(f"Wrote {kept} preference pairs ({ties} ties dropped) "
          f"-> {config.PREFERENCES_JSONL}")


if __name__ == "__main__":
    main()
