#!/usr/bin/env bash
# Data-parallel launcher for gen_completions.py.
#
# Each shard writes its own parquet; merge into completions.parquet at end.
#
# Usage:
#   run_gen.sh <model_path> <prompts_parquet> <out_dir> <num_shards>
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <model_path> <prompts_parquet> <out_dir> <num_shards>"
  exit 1
fi

MODEL="$1"; PROMPTS="$2"; OUT_DIR="$3"; NUM_SHARDS="$4"
PY="${PY:-python}"

mkdir -p "$OUT_DIR"

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
  out="$OUT_DIR/shard-$i.parquet"
  log="$OUT_DIR/shard-$i.log"
    nohup "$PY" "$(dirname "$0")/gen_completions.py" "$MODEL" "$PROMPTS" "$out" \
      --shard "$i" --num-shards "$NUM_SHARDS" \
      > "$log" 2>&1 &
  pids+=($!)
done

echo ">> launched pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
if [ "$fail" -ne 0 ]; then
  echo ">> ONE OR MORE SHARDS FAILED"
  for ((i=0; i<NUM_SHARDS; i++)); do
    echo "=== shard $i tail ==="
    tail -10 "$OUT_DIR/shard-$i.log"
  done
  exit 1
fi

echo ">> merging shards"
"$PY" - <<PY
import pandas as pd
from pathlib import Path
out = Path("$OUT_DIR")
dfs = [pd.read_parquet(out / f"shard-{i}.parquet") for i in range($NUM_SHARDS)]
merged = pd.concat(dfs, ignore_index=True)
merged.to_parquet(out / "completions.parquet", index=False)
print(f"merged {len(merged)} rows into {out / 'completions.parquet'}")
PY

echo ">> done"
