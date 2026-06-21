"""Thin Gemini wrapper: retries, JSON parsing, and a tiny on-disk cache so reruns
don't re-bill identical prompts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from . import config

_CACHE_DIR = config.RESULTS / ".llm_cache"


class Gemini:
    def __init__(self, model: str, *, temperature: float = 0.7,
                 use_cache: bool = True):
        self.model = model
        self.temperature = temperature
        self.use_cache = use_cache
        self.client = genai.Client(api_key=config.require_gemini_key())
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return _CACHE_DIR / f"{self.model}_{h}.txt"

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: Optional[float] = None, max_retries: int = 5) -> str:
        temp = self.temperature if temperature is None else temperature
        cache_key = f"{system}||{temp}||{prompt}"
        if self.use_cache:
            cp = self._cache_path(cache_key)
            if cp.exists():
                return cp.read_text()

        cfg = types.GenerateContentConfig(
            temperature=temp,
            system_instruction=system,
        )
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                text = (resp.text or "").strip()
                if self.use_cache:
                    self._cache_path(cache_key).write_text(text)
                return text
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
                time.sleep(min(2 ** attempt, 30))  # exponential backoff for 429s
        raise RuntimeError(f"Gemini failed after {max_retries} retries: {last_err}")

    def generate_json(self, prompt: str, *, system: Optional[str] = None,
                      temperature: float = 0.0) -> dict:
        """Generate and parse strict JSON, tolerating ```json fences."""
        raw = self.generate(prompt, system=system, temperature=temperature)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
