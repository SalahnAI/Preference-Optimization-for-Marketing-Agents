"""Phase 1 — build a table of campaign metrics that feed the recommendation prompt.

Two sources, same output schema (data/processed/campaigns.parquet):

  * synthetic  (default) — realistic, correlated ad metrics. Runs instantly, no
    download. CTR is modeled per channel.
  * criteo                — CTR is sourced *empirically* from the Criteo Display
    Advertising Challenge (real click labels, aggregated into pseudo-campaigns);
    campaign economics (budget/CPC/conversions) are then modeled the same way as
    synthetic. So the realistic part — the CTR distribution — comes from real
    ad-serving data. Stream the data in first (see scripts/get_criteo.sh).

Both share `_assemble()`, so the output schema is identical and the only
difference is where CTR comes from.

Output columns: campaign_id, channel, objective, ctr, cpc, budget, impressions,
clicks, conversions, cpa, roas.

    python -m marketing_agent.data_prep --source synthetic --n 400
    python -m marketing_agent.data_prep --source criteo --n 400
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

CHANNELS = ["Search", "Display", "Social", "Video", "Shopping"]
OBJECTIVES = ["Awareness", "Traffic", "Conversions", "Retargeting"]

# Conversion rate by objective — used to model conversions on top of CTR.
BASE_CVR = {"Conversions": 0.06, "Retargeting": 0.09,
            "Traffic": 0.02, "Awareness": 0.008}


def _round(df: pd.DataFrame) -> pd.DataFrame:
    df["ctr"] = df["ctr"].round(4)
    df["cpc"] = df["cpc"].round(2)
    df["cpa"] = df["cpa"].round(2)
    df["roas"] = df["roas"].round(2)
    df["budget"] = df["budget"].round(0).astype(int)
    return df


def _assemble(channel: np.ndarray, objective: np.ndarray, ctr: np.ndarray,
              rng: np.random.Generator) -> pd.DataFrame:
    """Given a CTR per campaign (modeled or empirical), model the rest of the
    funnel economics. Shared by both sources so the schema can't drift."""
    n = len(ctr)
    ctr = np.clip(ctr, 0.0008, 0.25)

    budget = rng.lognormal(mean=8.2, sigma=0.8, size=n)  # ~$3.6k median
    cpc = np.clip(rng.gamma(2.5, 0.18, n), 0.05, 6.0)

    clicks = (budget / cpc).astype(int)
    impressions = (clicks / ctr).astype(int)

    cvr = np.array([np.clip(rng.gamma(2.0, BASE_CVR[o] / 2.0), 0.001, 0.4)
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


def make_synthetic(n: int, seed: int = 7) -> pd.DataFrame:
    """CTR modeled per channel (Search clicks more than Display)."""
    rng = np.random.default_rng(seed)
    channel = rng.choice(CHANNELS, n)
    objective = rng.choice(OBJECTIVES, n)

    base_ctr = {"Search": 0.045, "Shopping": 0.030, "Social": 0.012,
                "Video": 0.008, "Display": 0.006}
    ctr = np.array([rng.gamma(2.0, base_ctr[c] / 2.0) for c in channel])
    return _assemble(channel, objective, ctr, rng)


def make_from_criteo(n: int, seed: int = 7, max_rows: int = 1_500_000,
                     min_impr: int = 200, target_ctr_mean: float = 0.02) -> pd.DataFrame:
    """Empirical per-segment click propensity from Criteo, then synthetic economics.

    Each pseudo-campaign is a bucket of Criteo rows sharing one categorical value
    (an anonymized segment). The *relative* click propensity across buckets is
    real signal. But the Criteo DAC negatives are subsampled, so its raw positive
    rate (~25%) is not a production CTR — we deflate it to `target_ctr_mean`
    (default 2%) while preserving the per-segment ratios. We auto-pick the
    categorical column whose bucketing yields enough campaigns of >= min_impr rows.

    Expects data/raw/train.txt (tab-separated: col0 = click label, I1-13 numeric,
    C1-26 categorical hashes).
    """
    path = config.DATA_RAW / "train.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/get_criteo.sh to stream it in, "
            "or use --source synthetic."
        )
    rng = np.random.default_rng(seed)
    cat_cols = [f"C{i}" for i in range(1, 27)]
    names = ["label"] + [f"I{i}" for i in range(1, 14)] + cat_cols
    df = pd.read_csv(path, sep="\t", names=names, usecols=["label"] + cat_cols,
                     nrows=max_rows, dtype={c: "category" for c in cat_cols})

    # Pick the categorical whose buckets best match the requested campaign count.
    best_col, best_eligible = cat_cols[0], -1
    for c in cat_cols:
        sizes = df.groupby(c, observed=True)["label"].size()
        eligible = int((sizes >= min_impr).sum())
        if eligible >= n:
            best_col = c
            break
        if eligible > best_eligible:
            best_col, best_eligible = c, eligible

    g = (df.groupby(best_col, observed=True)
           .agg(impressions=("label", "size"), clicks=("label", "sum")))
    g = (g[g["impressions"] >= min_impr]
         .sort_values("impressions", ascending=False).head(n).reset_index(drop=True))
    if len(g) < n:
        print(f"WARNING: only {len(g)} pseudo-campaigns available from column "
              f"{best_col} (wanted {n}). Lower --min_impr or raise --max_rows.")

    emp = (g["clicks"] / g["impressions"]).to_numpy()
    # Deflate the subsampled positive rate to a realistic CTR band, preserving
    # the relative per-segment propensity (ratios are unchanged by a linear scale).
    ctr = emp * (target_ctr_mean / emp.mean())
    m = len(g)
    channel = rng.choice(CHANNELS, m)
    objective = rng.choice(OBJECTIVES, m)
    out = _assemble(channel, objective, ctr, rng)
    print(f"Criteo: bucketed by {best_col}, {len(df):,} rows -> {m} campaigns; "
          f"raw CTR mean={emp.mean():.3f} -> rescaled to {out['ctr'].mean():.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build campaign metrics table.")
    ap.add_argument("--source", choices=["synthetic", "criteo"], default="synthetic")
    ap.add_argument("--n", type=int, default=400, help="number of campaigns")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_rows", type=int, default=1_500_000,
                    help="criteo: rows to read from train.txt")
    ap.add_argument("--min_impr", type=int, default=200,
                    help="criteo: min rows per pseudo-campaign")
    ap.add_argument("--target_ctr", type=float, default=0.02,
                    help="criteo: realistic CTR band to deflate the raw rate into")
    args = ap.parse_args()

    if args.source == "synthetic":
        df = make_synthetic(args.n, args.seed)
    else:
        df = make_from_criteo(args.n, args.seed, args.max_rows, args.min_impr,
                              args.target_ctr)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CAMPAIGNS_PARQUET, index=False)
    print(f"Wrote {len(df)} campaigns ({args.source}) -> {config.CAMPAIGNS_PARQUET}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
