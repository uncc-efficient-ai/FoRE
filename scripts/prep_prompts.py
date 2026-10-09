"""
Build the prompt set for drafter training data generation.

Mix, taking the first qualifying prompts in stream order:
- 150K from Nemotron-Post-Training-Dataset-v2, chat split
-  50K from its code split
-  30K from its math split
- every qualifying prompt of CodeAlpaca-20k

A Nemotron prompt is the first user message of a conversation; a CodeAlpaca prompt is the
instruction, followed by the input when there is one. Prompts are kept when they are
30-6000 characters long, a tokenizer-agnostic length proxy.

Output: a parquet with columns [id, source, prompt].
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

NEMOTRON = "nvidia/Nemotron-Post-Training-Dataset-v2"
CODE_ALPACA = "sahil2801/CodeAlpaca-20k"
# The dataset revisions the prompt pool was built from; selection is first-N in stream
# order.
NEMOTRON_REV = "5c89e01dd720ae0f4058445ed49c5fb68a03c76e"
CODE_ALPACA_REV = "152bb5e9a29651266b018106053980070a0521a1"

MIX = {
    "nemotron_chat": ("chat", 150_000),
    "nemotron_code": ("code", 50_000),
    "nemotron_math": ("math", 30_000),
}

MIN_CHARS = 30
MAX_CHARS = 6000


def extract_nemotron_user_prompt(ex):
    msgs = ex.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            return content if content else None
    return None


def pull_nemotron_split(split_name: str, target_n: int, source_tag: str):
    ds = load_dataset(NEMOTRON, data_files=f"data/{split_name}-*.parquet", split="train", streaming=True, revision=NEMOTRON_REV)
    out = []
    seen = 0
    for ex in ds:
        seen += 1
        p = extract_nemotron_user_prompt(ex)
        if p is None or not (MIN_CHARS <= len(p) <= MAX_CHARS):
            continue
        out.append({"source": source_tag, "prompt": p})
        if len(out) >= target_n:
            break
    print(f"  {source_tag}: kept {len(out)}/{seen} scanned")
    return out


def pull_codealpaca():
    ds = load_dataset(CODE_ALPACA, split="train", revision=CODE_ALPACA_REV)
    out = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        prompt = instr if not inp else f"{instr}\n\n{inp}"
        if not (MIN_CHARS <= len(prompt) <= MAX_CHARS):
            continue
        out.append({"source": "codealpaca", "prompt": prompt})
    print(f"  codealpaca: kept {len(out)}/{len(ds)} scanned")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_path", type=Path)
    args = ap.parse_args()

    rows = []
    for tag, (split, n) in MIX.items():
        print(f"pulling {tag} (target {n})")
        rows.extend(pull_nemotron_split(split, n, tag))
    print("pulling codealpaca")
    rows.extend(pull_codealpaca())

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"\nwrote {len(df)} prompts to {args.output_path}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
