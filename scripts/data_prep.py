"""Carve a regen completions JSONL (id/source/conversations/status, as written by
scripts/regenerate_train_data.py) into the train JSONL that D-PACE's
build_eagle3_dataset expects (id + conversations only) and a held-out prompts
parquet (id/prompt/completion). The held-out
split is deterministic so train data never overlaps val/test.

Usage:
  # write train JSONL excluding a seeded held-out set, dump held-out rows
  data_prep.py regen-100000-out.jsonl train.jsonl \
      --holdout-n 2000 --holdout-seed 7 --holdout-out holdout.parquet

  # or exclude an explicit id list (e.g. a held-out split's ids)
  data_prep.py regen-100000-out.jsonl train.jsonl --exclude-ids holdout_ids.txt
"""
import argparse
import json
import random

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_jsonl")
    ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout-n", type=int, default=None, help="seeded held-out size to exclude from out_jsonl")
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument("--holdout-out", default=None, help="parquet to write the held-out rows (id/prompt/completion)")
    ap.add_argument("--exclude-ids", default=None, help="text file of ids (one per line) to also exclude")
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "success":
                continue
            conv = r["conversations"]
            if len(conv) != 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
                raise SystemExit(f"id {r['id']}: expected a single user->assistant pair, got {[t['role'] for t in conv]}")
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    n_prompts = len({r["conversations"][0]["content"] for r in rows})
    if n_prompts != len(rows):
        raise SystemExit(f"{len(rows) - n_prompts} duplicate prompts — input should be the deduped regen output")

    holdout_ids = set()
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            holdout_ids |= {line.strip() for line in f if line.strip()}
    if args.holdout_n:
        held = random.Random(args.holdout_seed).sample(rows, args.holdout_n)
        holdout_ids |= {str(r["id"]) for r in held}
        if args.holdout_out:
            pd.DataFrame(
                {
                    "id": [str(r["id"]) for r in held],
                    "prompt": [r["conversations"][0]["content"] for r in held],
                    "completion": [r["conversations"][1]["content"] for r in held],
                }
            ).to_parquet(args.holdout_out)
            print(f"wrote {len(held)} held-out rows -> {args.holdout_out}")

    n = 0
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            if str(r["id"]) in holdout_ids:
                continue
            f.write(json.dumps({"id": str(r["id"]), "conversations": r["conversations"]}) + "\n")
            n += 1
    assert n + len(holdout_ids) >= len(rows), "carve lost rows"
    print(f"wrote {n} train conversations -> {args.out_jsonl} (excluded {len(holdout_ids)} held-out ids)")


if __name__ == "__main__":
    main()
