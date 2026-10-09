"""Build a nested, stratified regen input from the completions parquet.

Dedups prompts (keep first, matching data_prep.py), then lays the pool down in one
deterministic stratified order: the first-N rows for any N are a strict subset of the
first-M (M > N) and hold per-source proportions at every length.

Generation is split-agnostic: this emits one prompt set. The train/holdout split is the
consumer's job (data_prep.py --holdout-n/--holdout-seed, or --exclude-ids to pin a fixed
holdout across sizes).
"""
import argparse
import hashlib
import json

import pandas as pd


def frac_ranks(ids, seed):
    keyed = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
    n = len(keyed)
    return {i: (r + 0.5) / n for r, i in enumerate(keyed)}


def write_jsonl(df, path):
    with open(path, "w") as f:
        for r in df.itertuples(index=False):
            rid = r.id.item() if hasattr(r.id, "item") else r.id
            rec = {
                "id": rid,
                "source": r.source,
                "conversations": [{"role": "user", "content": r.prompt}],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="completions.parquet")
    ap.add_argument("--n", type=int, required=True, help="number of prompts to emit (prefix of the nested order)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sources", nargs="+", default=None, help="restrict pool to these sources (default full mix)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["id", "source", "prompt"])
    before = len(df)
    df = df.drop_duplicates(subset="prompt", keep="first").reset_index(drop=True)
    print(f"dedup: {before} -> {len(df)} unique prompts")

    pool = df if args.sources is None else df[df["source"].isin(args.sources)]

    ranks = {}
    for _, grp in pool.groupby("source"):
        ranks.update(frac_ranks(grp["id"].astype(str).tolist(), args.seed))
    pool = pool.assign(_fr=pool["id"].astype(str).map(ranks))
    pool = pool.sort_values(["_fr", "source", "id"], kind="stable")

    n = min(args.n, len(pool))
    sl = pool.iloc[:n]
    write_jsonl(sl, args.out)
    print(f"wrote {n} rows (of {len(pool)} available) -> {args.out}")
    print(sl["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
