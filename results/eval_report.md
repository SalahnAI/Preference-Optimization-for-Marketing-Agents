# Evaluation — base vs SFT vs DPO

Held-out set: 24 synthetic campaigns, `eval_seed=99` (disjoint from the preference
data). Inference ran **locally on an Apple M1 Max** (Qwen2.5-3B-Instruct in bf16 on
MPS — bitsandbytes 4-bit is CUDA-only, so the base was loaded full-precision instead
of 4-bit). Judge: **gemini-2.5-pro**, temperature 0, A/B position-swapped to cancel
order bias.

## Recommendation quality (LLM-as-judge, 1–5)

| Model | Actionability | Specificity | Business Value | Overall |
|---|---|---|---|---|
| base | 2.62 | 1.46 | 1.75 | **1.94** |
| sft  | 3.50 | 2.83 | 3.79 | **3.37** |
| dpo  | 4.46 | 2.88 | 3.12 | **3.49** |

## Pairwise preference win rate (position-swapped, n=24)

| Comparison | Win rate | W–L–T |
|---|---|---|
| SFT over base | **0.958** | 23–1–0 |
| DPO over base | **1.000** | 24–0–0 |
| DPO over SFT  | **0.750** | 18–6–0 |

## Takeaways

- **SFT is the big jump.** Teaching the model the task format/style on the `chosen`
  responses lifts overall quality 1.94 → 3.37 and wins 23/24 head-to-head vs base.
- **DPO adds a real, smaller lift on top of SFT.** It wins 18/24 (75%) against the
  SFT model and pushes overall quality to 3.49. The gain is concentrated in
  **actionability** (3.50 → 4.46) — DPO sharpens the "concrete lever + quantified
  justification" behaviour the preference pairs rewarded.
- **One trade-off shows up:** DPO's *business_value* score dips vs SFT (3.79 → 3.12)
  while *specificity* is roughly flat. DPO also produced **more concise** outputs
  (~1.2k vs ~1.0k vs ~1.9k chars for dpo/sft/base) — it trims the base model's
  padding. Net overall is still positive, but the lift is actionability-driven, not
  uniform.

## Files

- `eval_base_vs_dpo.{json,md}` — headline DPO-vs-base
- `eval_base_vs_sft.{json,md}` — SFT-vs-base
- `eval_sft_vs_dpo.{json,md}` — DPO-vs-SFT (the marginal DPO lift)
- `preds_{base,sft,dpo}.jsonl` — raw model outputs on the held-out set
