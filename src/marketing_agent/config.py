"""Central paths and model config. Import this; don't hardcode paths elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PREFS = ROOT / "data" / "preferences"
RESULTS = ROOT / "results"

# Generated artifacts (stable filenames referenced across the pipeline)
CAMPAIGNS_PARQUET = DATA_PROCESSED / "campaigns.parquet"
RECOMMENDATIONS_JSONL = DATA_PROCESSED / "recommendations.jsonl"
PREFERENCES_JSONL = DATA_PREFS / "preferences.jsonl"

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Generation model (cheap, high free-tier limits) and judge model (a notch stronger).
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-pro")


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return GEMINI_API_KEY
