"""Phase 2 — generate candidate recommendations per campaign.

For each campaign we sample TWO candidates: one "strong" (low temperature, full
system prompt) and one "weak" (higher temperature, a watered-down system prompt
that invites generic advice). The contrast is what Phase 3 turns into chosen/
rejected pairs. Output: data/processed/recommendations.jsonl

    python -m marketing_agent.generate --limit 400
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from . import config
from .gemini_client import Gemini
from .prompts import SYSTEM_PROMPT, WEAK_SYSTEM, recommendation_prompt, weak_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate recommendation candidates.")
    ap.add_argument("--limit", type=int, default=None, help="cap #campaigns")
    args = ap.parse_args()

    df = pd.read_parquet(config.CAMPAIGNS_PARQUET)
    if args.limit:
        df = df.head(args.limit)

    gen = Gemini(config.GEN_MODEL)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    n = 0
    with config.RECOMMENDATIONS_JSONL.open("w") as f:
        for _, c in tqdm(df.iterrows(), total=len(df), desc="generate"):
            prompt = recommendation_prompt(c)
            strong = gen.generate(prompt, system=SYSTEM_PROMPT, temperature=0.3)
            weak = gen.generate(weak_prompt(c), system=WEAK_SYSTEM, temperature=1.0)
            f.write(json.dumps({
                "campaign_id": c["campaign_id"],
                "prompt": prompt,
                "strong": strong,
                "weak": weak,
            }) + "\n")
            n += 1
    print(f"Wrote {n} records -> {config.RECOMMENDATIONS_JSONL}")


if __name__ == "__main__":
    main()
