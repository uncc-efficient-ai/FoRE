import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

MAX_PROMPT_TOKENS = 1024


def has_chat_template(tokenizer) -> bool:
    """Post-trained models ship a chat template; base models don't."""
    return getattr(tokenizer, "chat_template", None) is not None


def apply_qwen3_non_thinking(tokenizer, prompts: list[str]) -> list[str]:
    """Format prompts with the Qwen3 chat template, thinking disabled."""
    out = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # injects empty <think></think> block
        )
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("prompts_path", type=Path)
    ap.add_argument("output_path", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="cap total prompts processed by this shard (debug)")
    args = ap.parse_args()

    prompts_df = pd.read_parquet(args.prompts_path)
    if args.num_shards > 1:
        prompts_df = prompts_df.iloc[args.shard::args.num_shards].reset_index(drop=True)
    if args.max_prompts is not None:
        prompts_df = prompts_df.iloc[:args.max_prompts].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    use_chat_template = has_chat_template(tok)
    if use_chat_template:
        print(f"shard {args.shard}: chat template detected -> applying with enable_thinking=False")
        templated = apply_qwen3_non_thinking(tok, prompts_df["prompt"].tolist())
    else:
        print(f"shard {args.shard}: no chat template (base model) -> raw prompt continuation")
        templated = prompts_df["prompt"].tolist()

    # Tokenize the (possibly-templated) prompts to count tokens for filter + max_tokens budget.
    tokenized = tok(templated, add_special_tokens=False)
    prompt_token_counts = [len(ids) for ids in tokenized["input_ids"]]

    keep_mask = [c <= MAX_PROMPT_TOKENS for c in prompt_token_counts]
    kept_df = prompts_df.loc[keep_mask].reset_index(drop=True)
    dropped = sum(1 for k in keep_mask if not k)
    print(f"shard {args.shard}: {len(kept_df)} prompts kept, {dropped} dropped (>{MAX_PROMPT_TOKENS} toks)")

    rows = []
    for row in kept_df.itertuples():
        rows.append({
            "id": int(row.id),
            "source": row.source,
            "prompt": row.prompt,         # original user-content prompt (no template)
            "templated": use_chat_template,
        })

    out_df = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)
    print(f"wrote {len(out_df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
