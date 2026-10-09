#!/usr/bin/env bash
# Regenerate training data: one SGLang server per GPU, one client driving the full prompt set
# through all of them. Resumable by design (id-keyed --resume) — safe to re-run after an
# interruption; it skips completed ids and refills the servers.
#
# Prereqs:
#   - env built:   scripts/build_sglang_env.sh   (-> $ENV_DIR)
#   - prompts built: scripts/build_regen_input.py --n N --out $INPUT
#   - model at $MODEL
# If servers from a previous run are still up, kill them first (ports would collide),
# using the server PIDs that run printed ("server gpu=... pid=...").
#
#   bash scripts/run_regen.sh
# Overridable env: ENV_DIR MODEL NGPU BASE_PORT CONCURRENCY MEM_FRAC INPUT OUTPUT LOGDIR
set -euo pipefail

ENV_DIR="${ENV_DIR:-sglang-env}"
PY="$ENV_DIR/bin/python"
MODEL="${MODEL:-models/Qwen3.5-9B}"
NAME="$(basename "$MODEL")"
NGPU="${NGPU:-8}"
BASE_PORT="${BASE_PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-64}"          # per server; ~= the server's max_running_requests
THINK_FLAG="${THINK_FLAG:---no-thinking}"
MEM_FRAC="${MEM_FRAC:-0.8}"
INPUT="${INPUT:-data/regen-100000.jsonl}"
OUTPUT="${OUTPUT:-data/regen-100000-out.jsonl}"
LOGDIR="${LOGDIR:-.}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -f "$INPUT" ] || { echo "missing input: $INPUT  (build: $PY $SCRIPT_DIR/build_regen_input.py --n N --out $INPUT)"; exit 1; }

# 1. one server per GPU. --disable-radix-cache is REQUIRED: Qwen3.5's mamba/linear-attn
# layers otherwise force MambaRadixCache + no_buffer, which disables SGLang's overlap
# scheduler and stalls the server at full CPU with the GPU idle. The prompts are diverse,
# so the prefix cache would save little.
addrs=""
for i in $(seq 0 $((NGPU - 1))); do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$i nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$NAME" \
    --tp 1 --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$port" \
    --trust-remote-code --disable-radix-cache \
    > "$LOGDIR/sglang-srv-$i.log" 2>&1 &
  addrs="$addrs 127.0.0.1:$port"
  echo "server gpu=$i port=$port pid=$!"
done

# 2. wait for all servers ready
n=0
for _ in $(seq 1 60); do
  # || true: until a server fires up grep matches nothing and exits 1, which under
  # `set -e` + pipefail would end the script.
  n=$(grep -l "server is fired up" "$LOGDIR"/sglang-srv-*.log 2>/dev/null | wc -l || true)
  [ "$n" -ge "$NGPU" ] && break
  sleep 5
done
echo "servers ready: $n/$NGPU"
[ "$n" -ge "$NGPU" ] || { echo "not all servers came up — see $LOGDIR/sglang-srv-*.log"; exit 1; }

# 3. raise the fd limit BEFORE the client. CONCURRENCY*NGPU concurrent connections
# (64*8=512) plus connection churn exceed the default soft limit of 1024 and wedge the
# client with EMFILE (Too many open files). The nohup'd client inherits this raised limit.
ulimit -n "$(ulimit -Hn)"
echo "fd soft limit: $(ulimit -Sn)"

# 4. one client, full prompt set, round-robined across all servers; thinking off;
# id-keyed --resume skips completed ids.
nohup "$PY" "$SCRIPT_DIR/regenerate_train_data.py" \
  --model "$NAME" --server-address $addrs \
  --concurrency "$CONCURRENCY" --temperature 0.7 --top-p 0.95 --max-tokens 3072 \
  $THINK_FLAG --resume \
  --input-file-path "$INPUT" --output-file-path "$OUTPUT" \
  > "$LOGDIR/regen-client.log" 2>&1 &
echo "client pid=$! -> $OUTPUT  (log: $LOGDIR/regen-client.log)"
