import argparse
import time

import torch

from .decode import generate, load


def main():
    ap = argparse.ArgumentParser(
        description="Generate with FoRE (MTP-guided early forking and draft refinement) on Qwen3.5."
    )
    ap.add_argument("--target", required=True, help="Qwen3.5-9B or Qwen3.5-4B checkpoint directory")
    ap.add_argument("--drafter", required=True, help="FoRE drafter checkpoint directory")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--k", type=int, default=8, help="K, the number of MTP-guided drafts")
    ap.add_argument(
        "--r", type=int, default=8,
        help="R, the number of additional drafts from draft refinement (0 for early forking alone)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--attention", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    args = ap.parse_args()

    target, drafter, mtp, tokenizer = load(args.target, args.drafter, args.attention)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids, steps = generate(
        target, drafter, mtp, input_ids, args.max_new_tokens, [tokenizer.eos_token_id], k=args.k, r=args.r
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    print(tokenizer.decode(ids[0], skip_special_tokens=True))
    print(
        f"\n{ids.shape[1]} tokens in {len(steps)} steps: "
        f"{sum(steps) / len(steps):.2f} tokens/step, {ids.shape[1] / seconds:.1f} tokens/s (includes prefill)"
    )
