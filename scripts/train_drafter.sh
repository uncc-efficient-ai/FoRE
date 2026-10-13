#!/usr/bin/env bash
# Train a FoRE drafter, skipping each step whose output exists. Variables: docs/training.md.
#
#   TARGET=models/Qwen3.5-9B DFLASH=models/Qwen3.5-9B-DFlash bash scripts/train_drafter.sh 9b
set -euo pipefail

size=${1:-}
case "$size" in
  9b) exclude_ids=(data/decontam-exclude-ids.txt data/holdout-ids-qwen3.5-9b.txt) ;;
  4b) exclude_ids=(data/holdout-ids-qwen3.5-4b.txt) ;;
  *) echo "usage: TARGET=... DFLASH=... $0 9b|4b" >&2; exit 2 ;;
esac
: "${TARGET:?set TARGET to the target model directory}"
: "${DFLASH:?set DFLASH to the released DFlash drafter directory}"
N=${N:-100000}
NGPU=${NGPU:-8}
NPROC=${NPROC:-4}
for v in N NGPU NPROC; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer, got '${!v}'" >&2; exit 2; }
done
root=$(cd "$(dirname "$0")/.." && pwd)
TARGET=$(realpath -m "$TARGET")
DFLASH=$(realpath -m "$DFLASH")
DATA=$(realpath -m "${DATA:-$root/data}")
RUNS=$(realpath -m "${RUNS:-$root/runs}")
ENV_DIR=$(realpath -m "${ENV_DIR:-$root/sglang-env}")

cd "$root"
run=(uv run --locked --extra flash-attn --extra train)
epochs=1
name=qwen3.5-$size
prompts=$DATA/prompts.parquet
pool=$DATA/pool
regen_in=$DATA/regen-$N.jsonl
regen_out=$DATA/regen-$N-$name-out.jsonl
train=$DATA/train-$N-$name.jsonl
out=$RUNS/fore-$name-$N
mkdir -p "$DATA" "$RUNS"

step() { echo; echo "== step $1"; }

# The final save is epoch_<epochs>_step_<N>; earlier saves carry a 0-based epoch. It counts
# once its weights file opens.
final_checkpoint() {
  local d
  for d in "$out"/epoch_"$epochs"_step_*; do
    [ -f "$d/config.json" ] && [ -f "$d/model.safetensors" ] || continue
    "${run[@]}" python -c 'import sys; from safetensors import safe_open; safe_open(sys.argv[1], "pt")' \
      "$d/model.safetensors" 2>/dev/null && { echo "$d"; return; }
  done
  return 0
}

step "1: prompt pool -> $prompts"
if [ -s "$prompts" ]; then echo "exists, skipped"; else
  "${run[@]}" python scripts/prep_prompts.py "$prompts.tmp"
  mv "$prompts.tmp" "$prompts"
fi

step "2: length filter -> $pool/completions.parquet"
if [ -s "$pool/completions.parquet" ]; then echo "exists, skipped"; else
  rm -rf "$pool.tmp"
  # Deduplication depends on the shard count.
  PY=python "${run[@]}" bash scripts/run_gen.sh \
    "$TARGET" \
    "$prompts" \
    "$pool.tmp" \
    7
  rm -rf "$pool"
  mv "$pool.tmp" "$pool"
fi

step "3: $N-prompt sample -> $regen_in"
if [ -s "$regen_in" ]; then echo "exists, skipped"; else
  "${run[@]}" python scripts/build_regen_input.py \
    --parquet "$pool/completions.parquet" \
    --n "$N" \
    --out "$regen_in.tmp"
  mv "$regen_in.tmp" "$regen_in"
fi

step "4: responses -> $regen_out"
# A partial output is resumed, so completion has its own marker.
if [ -e "$regen_out.done" ]; then echo "exists, skipped"; else
  if ! "$ENV_DIR/bin/python" -c 'import sglang' 2>/dev/null; then
    ENV_DIR="$ENV_DIR" bash scripts/build_sglang_env.sh
  fi
  mkdir -p "$DATA/logs-$name"
  WAIT=1 \
  ENV_DIR="$ENV_DIR" \
  MODEL="$TARGET" \
  NGPU="$NGPU" \
  INPUT="$regen_in" \
  OUTPUT="$regen_out" \
  LOGDIR="$DATA/logs-$name" \
    bash scripts/run_regen.sh
  touch "$regen_out.done"
fi

step "5: split -> $train"
if [ -s "$train" ]; then echo "exists, skipped"; else
  cat "${exclude_ids[@]}" > "$DATA/exclude-$name.txt"
  "${run[@]}" python scripts/data_prep.py \
    "$regen_out" \
    "$train.tmp" \
    --exclude-ids "$DATA/exclude-$name.txt"
  # Step 6's tokenized cache is keyed by the train file's path, not its contents.
  rm -rf "$DATA/cache"
  mv "$train.tmp" "$train"
fi

step "6: fine-tune -> $out"
final=$(final_checkpoint)
if [ -n "$final" ]; then echo "exists, skipped"; else
  mkdir -p "$out"
  inputs="TARGET=$TARGET DFLASH=$DFLASH"
  if [ -f "$out/inputs.txt" ] && [ "$(cat "$out/inputs.txt")" != "$inputs" ]; then
    echo "$out was started with $(cat "$out/inputs.txt"); set RUNS or remove it" >&2
    exit 1
  fi
  echo "$inputs" > "$out/inputs.txt"
  # An interrupted final save would otherwise be the checkpoint --resume picks.
  for d in "$out"/epoch_"$epochs"_step_*; do
    [ -d "$d" ] && { echo "removing incomplete $d"; rm -rf "$d"; }
  done
  "${run[@]}" torchrun \
    --standalone \
    --nproc-per-node "$NPROC" \
    -m fore.train.train_dflash \
    --target-model-path "$TARGET" \
    --target-model-backend hf \
    --init-from "$DFLASH" \
    --train-data-path "$train" \
    --block-size 16 \
    --max-length 3072 \
    --num-anchors 512 \
    --chat-template qwen3-nosys \
    --embedding-key model.language_model.embed_tokens.weight \
    --trust-remote-code \
    --num-epochs "$epochs" \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --loss-type dpace \
    --dpace-alpha 0.5 \
    --variable-prefix \
    --attention-backend flex_attention \
    --seed 42 \
    --log-interval 20 \
    --save-interval 1000 \
    --resume \
    --output-dir "$out" \
    --cache-dir "$DATA/cache"
  final=$(final_checkpoint)
  [ -n "$final" ] || { echo "no complete final checkpoint in $out" >&2; exit 1; }
fi

echo
echo "drafter: $final"
