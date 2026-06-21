"""Phase 5 — supervised fine-tuning (LoRA) on the CHOSEN recommendations.

SFT teaches the base model the task format + the "good" style, giving DPO a
sensible reference policy to improve on. Run on Colab (GPU):

    pip install -r training/requirements-train.txt
    python training/sft_train.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data data/preferences/preferences.jsonl \
        --out training/outputs/sft

Defaults are tuned for a free T4 (4-bit QLoRA). Swap --model for
meta-llama/Llama-3.2-3B-Instruct if you prefer Llama.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from prompts_shared import SYSTEM_PROMPT  # see note at bottom


def build_text(example, tokenizer):
    """Render one (prompt, chosen) pair as a chat-formatted training string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/preferences/preferences.jsonl")
    ap.add_argument("--out", default="training/outputs/sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda e: build_text(e, tok), remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=1024,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved SFT adapter -> {args.out}")


if __name__ == "__main__":
    main()
