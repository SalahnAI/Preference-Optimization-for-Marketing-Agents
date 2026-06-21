"""Phase 1 — build a table of campaign metrics that feed the recommendation prompt.

Two sources, same output schema (data/processed/campaigns.parquet):

  * synthetic  (default) — realistic, correlated ad metrics. Runs instantly, no
    download. Good enough because the agent only ever sees the *aggregated*
    metrics below, not raw impressions.
  * criteo                — aggregate the Criteo Display Advertising Challenge
    (label + 13 numeric features) into pseudo-campaigns. Use if you want to cite
    a public dataset. Drop train.txt in data/raw/ first.

Output columns: campaign_id, channel, objective, ctr, cpc, budget, impressions,
clicks, conversions, cpa, roas.

    python -m marketing_agent.data_prep --source synthetic --n 400
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

CHANNELS = ["Search", "Display", "Social", "Video", "Shopping"]
OBJECTIVES = ["Awareness", "Traffic", "Conversions", "Retargeting"]


def _round(df: pd.DataFrame) -> pd.DataFrame:
    df["ctr"] = df["ctr"].round(4)
    df["cpc"] = df["cpc"].round(2)
    df["cpa"] = df["cpa"].round(2)
    df["roas"] = df["roas"].round(2)
    df["budget"] = df["budget"].round(0).astype(int)
    return df


def make_synthetic(n: int, seed: int = 7) -> pd.DataFrame:
    """Correlated, plausible campaign metrics (not i.i.d. noise)."""
    rng = np.random.default_rng(seed)

    channel = rng.choice(CHANNELS, n)
    objective = rng.choice(OBJECTIVES, n)

    # CTR varies by channel; Search clicks more than Display.
    base_ctr = {"Search": 0.045, "Shopping": 0.030, "Social": 0.012,
                "Video": 0.008, "Display": 0.006}
    ctr = np.array([rng.gamma(2.0, base_ctr[c] / 2.0) for c in channel])
    ctr = np.clip(ctr, 0.0008, 0.25)

    budget = rng.lognormal(mean=8.2, sigma=0.8, size=n)  # ~$3.6k median
    cpc = np.clip(rng.gamma(2.5, 0.18, n), 0.05, 6.0)

    clicks = (budget / cpc).astype(int)
    impressions = (clicks / ctr).astype(int)

    # Conversion rate depends on objective.
    base_cvr = {"Conversions": 0.06, "Retargeting": 0.09,
                "Traffic": 0.02, "Awareness": 0.008}
    cvr = np.array([np.clip(rng.gamma(2.0, base_cvr[o] / 2.0), 0.001, 0.4)
                    for o in objective])
    conversions = np.maximum((clicks * cvr).astype(int), 0)

    cpa = np.where(conversions > 0, budget / np.maximum(conversions, 1), np.nan)
    aov = rng.uniform(25, 180, n)  # average order value
    roas = np.where(conversions > 0, (conversions * aov) / budget, 0.0)

    df = pd.DataFrame({
        "campaign_id": [f"C{1000 + i}" for i in range(n)],
        "channel": channel,
        "objective": objective,
        "ctr": ctr,
        "cpc": cpc,
        "budget": budget,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "cpa": cpa,
        "roas": roas,
    })
    return _round(df)


def make_from_criteo(n: int, seed: int = 7) -> pd.DataFrame:
    """Aggregate Criteo rows into pseudo-campaigns keyed by a categorical hash.

    Expects data/raw/train.txt (tab-separated, col 0 = click label, cols 1-13
    numeric, 14-39 categorical). We bucket rows by C1 to form "campaigns" and
    derive CTR from the label mean; budget/cpc are sampled to stay realistic.
    """
    path = config.DATA_RAW / "train.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Criteo Display Advertising Challenge "
            "dataset and place train.txt in data/raw/, or use --source synthetic."
        )
    rng = np.random.default_rng(seed)
    # Stream a manageable slice; the full file is ~11GB.
    cols = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    chunk = pd.read_csv(path, sep="\t", names=cols, nrows=2_000_000)

    grp = chunk.groupby("C1").agg(impressions=("label", "size"),
                                  clicks=("label", "sum")).reset_index()
    grp = grp[grp["impressions"] >= 500].head(n).reset_index(drop=True)

    ctr = np.clip(grp["clicks"] / grp["impressions"], 0.0008, 0.25).to_numpy()
    cpc = np.clip(rng.gamma(2.5, 0.18, len(grp)), 0.05, 6.0)
    clicks = grp["clicks"].to_numpy()
    budget = clicks * cpc
    cvr = np.clip(rng.gamma(2.0, 0.03, len(grp)), 0.001, 0.4)
    conversions = np.maximum((clicks * cvr).astype(int), 0)
    aov = rng.uniform(25, 180, len(grp))

    df = pd.DataFrame({
        "campaign_id": [f"C{1000 + i}" for i in range(len(grp))],
        "channel": rng.choice(CHANNELS, len(grp)),
        "objective": rng.choice(OBJECTIVES, len(grp)),
        "ctr": ctr,
        "cpc": cpc,
        "budget": budget,
        "impressions": grp["impressions"].to_numpy(),
        "clicks": clicks,
        "conversions": conversions,
        "cpa": np.where(conversions > 0, budget / np.maximum(conversions, 1), np.nan),
        "roas": np.where(conversions > 0, (conversions * aov) / budget, 0.0),
    })
    return _round(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build campaign metrics table.")
    ap.add_argument("--source", choices=["synthetic", "criteo"], default="synthetic")
    ap.add_argument("--n", type=int, default=400, help="number of campaigns")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = (make_synthetic if args.source == "synthetic" else make_from_criteo)(
        args.n, args.seed)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CAMPAIGNS_PARQUET, index=False)
    print(f"Wrote {len(df)} campaigns ({args.source}) -> {config.CAMPAIGNS_PARQUET}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
