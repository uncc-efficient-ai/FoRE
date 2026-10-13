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
# Overridable env: ENV_DIR MODEL NGPU BASE_PORT CONCURRENCY MEM_FRAC INPUT OUTPUT LOGDIR WAIT
# WAIT=1 waits for the client, then stops the servers. The client exits non-zero if any prompt
# failed; a rerun retries those prompts.
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
WAIT="${WAIT:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -f "$INPUT" ] || { echo "missing input: $INPUT  (build: $PY $SCRIPT_DIR/build_regen_input.py --n N --out $INPUT)"; exit 1; }

[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo "NGPU must be a positive integer, got '$NGPU'"; exit 1; }
if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
  IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
  IFS=, read -r -a gpus <<< "$(seq -s, 0 $((NGPU - 1)))"
fi
[ "${#gpus[@]}" -ge "$NGPU" ] || { echo "NGPU=$NGPU but only ${#gpus[@]} GPUs visible: ${gpus[*]:-none}"; exit 1; }

addrs=""
pids=""
if [ "$WAIT" = 1 ]; then
  # setsid puts each server's child processes in its group. With requests in flight a server
  # can take over 20 s to exit on SIGTERM, hence the wait before SIGKILL.
  stop() {
    kill $pids 2>/dev/null || true
    for _ in $(seq 30); do
      alive=0
      for p in $pids; do kill -0 -- "-$p" 2>/dev/null || kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && return
      sleep 1
    done
    for p in $pids; do kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; done
  }
  trap stop EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
fi

# Logs of an earlier, wider run would otherwise count as ready servers.
rm -f "$LOGDIR"/sglang-srv-*.log
# --disable-radix-cache is REQUIRED: Qwen3.5's mamba/linear-attn layers otherwise force
# MambaRadixCache + no_buffer, which disables SGLang's overlap scheduler and stalls the server
# at full CPU with the GPU idle. The prompts are diverse, so the prefix cache would save little.
for i in $(seq 0 $((NGPU - 1))); do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=${gpus[$i]} setsid nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$NAME" \
    --tp 1 --mem-fraction-static "$MEM_FRAC" --host 127.0.0.1 --port "$port" \
    --trust-remote-code --disable-radix-cache \
    > "$LOGDIR/sglang-srv-$i.log" 2>&1 &
  addrs="$addrs 127.0.0.1:$port"
  pids="$pids $!"
  echo "server gpu=${gpus[$i]} port=$port pid=$!"
done

for _ in $(seq 1 60); do
  n=0
  for i in $(seq 0 $((NGPU - 1))); do
    grep -q "server is fired up" "$LOGDIR/sglang-srv-$i.log" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -ge "$NGPU" ] && break
  sleep 5
done
echo "servers ready: $n/$NGPU"
[ "$n" -ge "$NGPU" ] || { echo "not all servers came up — see $LOGDIR/sglang-srv-*.log"; exit 1; }

# Raise the fd limit before the client starts: CONCURRENCY*NGPU concurrent connections
# (64*8=512) plus connection churn exceed the default soft limit of 1024 and wedge the
# client with EMFILE (Too many open files). The nohup'd client inherits the raised limit.
ulimit -n "$(ulimit -Hn)"
echo "fd soft limit: $(ulimit -Sn)"

nohup "$PY" "$SCRIPT_DIR/regenerate_train_data.py" \
  --model "$NAME" --server-address $addrs \
  --concurrency "$CONCURRENCY" --temperature 0.7 --top-p 0.95 --max-tokens 3072 \
  $THINK_FLAG --resume --retry-errors \
  --input-file-path "$INPUT" --output-file-path "$OUTPUT" \
  > "$LOGDIR/regen-client.log" 2>&1 &
client=$!
echo "client pid=$client -> $OUTPUT  (log: $LOGDIR/regen-client.log)"
if [ "$WAIT" = 1 ]; then
  pids="$pids $client"
  wait "$client"
fi
