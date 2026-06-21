"""Single source of truth for the system prompt, importable from training scripts
without installing the package (Colab just clones the repo).

Re-exports SYSTEM_PROMPT from src/marketing_agent/prompts.py so SFT, DPO, and the
data pipeline can never drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marketing_agent.prompts import SYSTEM_PROMPT  # noqa: E402,F401
