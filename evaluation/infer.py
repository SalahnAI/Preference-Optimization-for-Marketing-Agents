"""Generate recommendations from a trained model on a HELD-OUT campaign set.

Run once per model variant on Colab (base / sft / dpo), then judge the outputs
locally with run_eval.py. The held-out campaigns use a different seed than the
training data so we never evaluate on campaigns seen during preference building.

    # base (no adapter)
    python evaluation/infer.py --base Qwen/Qwen2.5-3B-Instruct --tag base
    # sft / dpo (with adapter)
    python evaluation/infer.py --base Qwen/Qwen2.5-3B-Instruct \
        --adapter training/outputs/dpo --tag dpo

Writes results/preds_<tag>.jsonl with {campaign_id, metrics_block, prompt, output}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from marketing_agent.data_prep import make_synthetic  # noqa: E402
from marketing_agent.prompts import (  # noqa: E402
    SYSTEM_PROMPT, campaign_block, recommendation_prompt)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for base model")
    ap.add_argument("--tag", required=True, help="base | sft | dpo")
    ap.add_argument("--n_eval", type=int, default=60)
    ap.add_argument("--eval_seed", type=int, default=99)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load the base in 4-bit, exactly as training did, so the comparison is fair
    # and PEFT injects the adapter via the bitsandbytes backend (avoids the
    # torchao dispatch path that fails on older torchao).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    campaigns = make_synthetic(args.n_eval, seed=args.eval_seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"preds_{args.tag}.jsonl"

    with out_path.open("w") as f:
        for _, c in campaigns.iterrows():
            prompt = recommendation_prompt(c)
            chat = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=400,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            f.write(json.dumps({
                "campaign_id": c["campaign_id"],
                "metrics_block": campaign_block(c),
                "prompt": prompt,
                "output": text,
            }) + "\n")
    print(f"Wrote {len(campaigns)} preds -> {out_path}")


if __name__ == "__main__":
    main()
