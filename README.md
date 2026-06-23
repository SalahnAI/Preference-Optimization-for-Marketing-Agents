# Preference Optimization for Marketing Agents

Fine-tune a small open LLM to give better marketing-campaign recommendations,
using **Direct Preference Optimization (DPO)** on top of supervised fine-tuning —
and benchmark the result against the base model, the SFT model, and frontier
APIs on **recommendation quality, latency, and cost**.

> The point of this repo is not "I called an LLM." It's a full
> **evaluation-and-alignment pipeline**: generate data → build preferences →
> SFT → DPO → measure the lift.

## Pipeline

```
Campaign metrics      data_prep.py         (Phase 1)
        ↓
Recommendations       generate.py          (Phase 2)  — strong vs weak candidates
        ↓
Preference pairs      build_preferences.py (Phase 3)  — Gemini judge → chosen/rejected
        ↓
SFT (LoRA)            training/sft_train.py (Phase 5)
        ↓
DPO                   training/dpo_train.py (Phase 6)
        ↓
Evaluation            evaluation/           (Phase 7)  — win rate, quality, latency, cost
```

## Repo layout

```
data/
  raw/            # (optional) Criteo train.txt — gitignored
  processed/      # campaigns.parquet, recommendations.jsonl
  preferences/    # preferences.jsonl  ← the core artifact (committed)
src/marketing_agent/   # data + generation + preference building (runs locally)
training/         # sft_train.py, dpo_train.py  (run on Colab GPU)
evaluation/       # infer.py (Colab), run_eval.py (local)
results/          # eval_report.{json,md}, prediction dumps
notebooks/        # Colab driver notebooks
```

## Quickstart

### 1. Local pipeline (data → preferences). No GPU needed.

The Gemini SDK needs Python ≥3.10, so use a venv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
export PYTHONPATH=src

# Phase 1 — build campaign metrics (synthetic by default; --source criteo also works)
python -m marketing_agent.data_prep --source synthetic --n 400

# Phase 2 — generate strong/weak recommendation candidates
python -m marketing_agent.generate

# Phase 3 — judge pairs into a DPO dataset (data/preferences/preferences.jsonl)
python -m marketing_agent.build_preferences
```

> Prefer Colab for everything? [`notebooks/colab_data_pipeline.ipynb`](notebooks/colab_data_pipeline.ipynb)
> runs Phases 1-3 with no local setup.

### 2. Training (free GPU notebook)

One notebook runs SFT → DPO → inference → judging in order — pick your platform:

- **SageMaker Studio Lab** (free, no card, T4 / 4h sessions): [`notebooks/studiolab_sft_dpo_eval.ipynb`](notebooks/studiolab_sft_dpo_eval.ipynb)
- **Google Colab** (free T4): [`notebooks/colab_sft_dpo_eval.ipynb`](notebooks/colab_sft_dpo_eval.ipynb)

Or from a shell on any GPU box:

```bash
pip install -r training/requirements-train.txt
python training/sft_train.py --model Qwen/Qwen2.5-3B-Instruct
python training/dpo_train.py --base Qwen/Qwen2.5-3B-Instruct --sft_adapter training/outputs/sft
```

4-bit QLoRA fits a 3B model on a free T4. Swap in
`meta-llama/Llama-3.2-3B-Instruct` if preferred.

> **AWS note:** the 12-month *free tier* has **no GPU** — only Studio Lab (above) is
> truly free. Real AWS *credits* (Activate / promo) can run an EC2 `g5.xlarge` or a
> SageMaker job for ~$1-2 total, but a fresh account needs a **G-instance quota
> increase** first (Service Quotas → "Running On-Demand G instances").

### 3. Evaluation

```bash
# On Colab: dump predictions for each variant on a held-out campaign set
python evaluation/infer.py --tag base
python evaluation/infer.py --adapter training/outputs/sft --tag sft
python evaluation/infer.py --adapter training/outputs/dpo --tag dpo

# Locally: judge them (only needs Gemini)
python evaluation/run_eval.py --a results/preds_base.jsonl --b results/preds_dpo.jsonl
```

## Methodology

- **Preferences** come from a *strong vs weak* generation contrast (different
  system prompt + temperature), labelled by a Gemini judge with **A/B position
  swapping** to cancel order bias. Ties are dropped.
- **SFT** trains on the `chosen` responses to fix the task format and give DPO a
  sane reference policy.
- **DPO** (TRL `DPOTrainer`, β=0.1) starts from the SFT adapter; the reference
  policy is the same model with the adapter disabled.
- **Evaluation** holds out a campaign set generated with a different seed, so we
  never judge on campaigns used to build preferences. The same judge prompt is
  reused, scoring Actionability / Specificity / Business Value plus a pairwise
  win rate.

## Results

Populated by `evaluation/run_eval.py` → `results/eval_report.md`. Run below was
done **locally on an Apple M1 Max** (Qwen2.5-3B-Instruct, bf16/MPS — no 4-bit) over a
24-campaign held-out set (`eval_seed=99`), judged by gemini-2.5-pro.

**Recommendation quality (LLM-as-judge, 1–5):**

| Model | Actionability | Specificity | Business Value | Overall |
|---|---|---|---|---|
| base | 2.62 | 1.46 | 1.75 | **1.94** |
| sft  | 3.50 | 2.83 | 3.79 | **3.37** |
| dpo  | 4.46 | 2.88 | 3.12 | **3.49** |

**Pairwise preference win rate (position-swapped, n=24):** SFT over base **0.958**
(23–1), DPO over base **1.000** (24–0), DPO over SFT **0.750** (18–6).

## Lessons learned

- **SFT does the heavy lifting**: overall 1.94 → 3.37 and a 23/24 win vs base —
  teaching the task format/style is most of the gain.
- **DPO adds a real but smaller lift on top of SFT** (3.37 → 3.49, 18/24 head-to-head
  win), concentrated in **actionability** (3.50 → 4.46) — exactly the axis the
  strong-vs-weak preference pairs rewarded. It also makes outputs more concise.
- **Trade-off**: DPO's *business_value* dips vs SFT (3.79 → 3.12), so the lift is
  actionability-driven rather than uniform — worth watching if you re-weight the
  preference signal.
- **Judge reliability**: gemini-2.5-pro on the free tier is the bottleneck (~15–20 s
  per call); local Apple-Silicon inference (~6.5 min per variant for 24 campaigns)
  was actually the faster half of the loop.

## Dataset

The agent only ever sees **aggregated campaign metrics** (CTR, CPC, budget,
ROAS…), so the data layer just needs to produce a realistic table of those. Two
interchangeable sources, same output schema:

- **`--source synthetic`** (default) — correlated, plausible metrics: CTR modeled
  per channel, CVR per objective, funnel economics derived. Instant, no download.
- **`--source criteo`** — **CTR is sourced empirically** from the public
  [Criteo Display Advertising Challenge](https://ailab.criteo.com/display-advertising-challenge-criteo/)
  (real click labels, aggregated into per-segment pseudo-campaigns); the campaign
  economics are then modeled the same way as synthetic. So the realistic part —
  the CTR distribution — comes from real ad-serving data.

```bash
# Stream just what's needed (~0.9 GB, not the full 4.58 GB archive)
scripts/get_criteo.sh                 # -> data/raw/train.txt (first 1.5M rows)
PYTHONPATH=src python -m marketing_agent.data_prep --source criteo --n 400
```

Both paths share `_assemble()` in [data_prep.py](src/marketing_agent/data_prep.py),
so switching source never changes the downstream pipeline.
