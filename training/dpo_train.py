"""Phase 6 — DPO on top of the SFT adapter.

DPO nudges the policy toward `chosen` and away from `rejected` without a reward
model. We start from the SFT adapter so the reference policy already speaks the
task format. Run on Colab (GPU):

    python training/dpo_train.py \
        --base Qwen/Qwen2.5-3B-Instruct \
        --sft_adapter training/outputs/sft \
        --data data/preferences/preferences.jsonl \
        --out training/outputs/dpo
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from prompts_shared import SYSTEM_PROMPT


def format_for_dpo(example, tokenizer):
    """DPOTrainer wants prompt/chosen/rejected as chat-templated strings."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": example["prompt"]}],
        tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft_adapter", default="training/outputs/sft")
    ap.add_argument("--data", default="data/preferences/preferences.jsonl")
    ap.add_argument("--out", default="training/outputs/dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    # Load SFT adapter as the trainable starting point. TRL builds the frozen
    # reference policy internally (ref = policy with adapter disabled).
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda e: format_for_dpo(e, tok),
                remove_columns=[c for c in ds.column_names
                                if c not in ("prompt", "chosen", "rejected")])

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,
        max_prompt_length=640,
        report_to="none",
    )
    # New LoRA on top so DPO trains its own delta; ref = this disabled.
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
