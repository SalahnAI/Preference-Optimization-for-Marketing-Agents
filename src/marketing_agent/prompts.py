"""All prompt templates live here so training/eval reuse the exact same text."""
from __future__ import annotations

from typing import Mapping

# The instruction the agent is trained and evaluated on. SFT/DPO targets must be
# generated against THIS exact prompt so the model learns the right mapping.
SYSTEM_PROMPT = (
    "You are a senior performance-marketing analyst. Given a campaign's metrics, "
    "give concrete, prioritized recommendations a marketer can act on this week. "
    "Be specific and quantitative; avoid generic advice."
)


def campaign_block(c: Mapping) -> str:
    """Render one campaign row as the metrics block shown to the model."""
    return (
        f"Channel: {c['channel']}\n"
        f"Objective: {c['objective']}\n"
        f"CTR: {c['ctr']:.2%}\n"
        f"CPC: ${c['cpc']:.2f}\n"
        f"Budget: ${int(c['budget']):,}\n"
        f"Impressions: {int(c['impressions']):,}\n"
        f"Clicks: {int(c['clicks']):,}\n"
        f"Conversions: {int(c['conversions']):,}\n"
        f"CPA: {'$%.2f' % c['cpa'] if c['cpa'] == c['cpa'] else 'n/a'}\n"
        f"ROAS: {c['roas']:.2f}x"
    )


def recommendation_prompt(c: Mapping) -> str:
    return (
        "Campaign metrics:\n\n"
        f"{campaign_block(c)}\n\n"
        "Provide exactly 3 actionable recommendations, numbered 1-3. "
        "Each: one bold lever, then a quantitative justification tied to the "
        "metrics above (~2 sentences)."
    )


# Pairwise judge used to BUILD preferences (Phase 3) and to EVALUATE (Phase 7).
JUDGE_SYSTEM = (
    "You are a strict evaluator of marketing recommendations. Judge on "
    "Actionability, Specificity, and Business Value. Reward quantitative reasoning "
    "tied to the metrics; penalize vague or generic advice."
)


def pairwise_judge_prompt(metrics_block: str, a: str, b: str) -> str:
    return (
        f"Campaign metrics:\n{metrics_block}\n\n"
        f"--- Recommendation A ---\n{a}\n\n"
        f"--- Recommendation B ---\n{b}\n\n"
        "Which recommendation is better? Reply with strict JSON only: "
        '{\"winner\": \"A\" | \"B\" | \"tie\", \"reason\": \"<one sentence>\"}'
    )


def score_judge_prompt(metrics_block: str, rec: str) -> str:
    return (
        f"Campaign metrics:\n{metrics_block}\n\n"
        f"Recommendation:\n{rec}\n\n"
        "Score 1-5 on each axis. Reply with strict JSON only: "
        '{\"actionability\": int, \"specificity\": int, \"business_value\": int, '
        '\"reason\": \"<one sentence>\"}'
    )
