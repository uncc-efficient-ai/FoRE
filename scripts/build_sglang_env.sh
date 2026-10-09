#!/usr/bin/env bash
# Isolated SGLang environment that serves the target for response regeneration. SGLang
# pins torch 2.11 and the trainer torch 2.12, so the two cannot share an environment.
# Built with: sglang 0.5.6.post3.dev6105+g53ed34cb8 / torch 2.11.0+cu130, CUDA 13.
#
#   bash scripts/build_sglang_env.sh
# Overridable: ENV_DIR, SGLANG_REF, MAX_JOBS, TMPDIR.
set -euo pipefail

ENV_DIR="${ENV_DIR:-sglang-env}"
SGLANG_REF="${SGLANG_REF:-53ed34cb8}"          # pinned main commit; main moves, this doesn't
PROTOC_VER="${PROTOC_VER:-35.0}"
TMPDIR="${TMPDIR:-tmp}"; mkdir -p "$TMPDIR"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Rust toolchain — the sglang-grpc crate builds via prost.
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi

# protoc — prost-build needs the binary (not the pip package); installed into ~/.local.
if ! command -v protoc >/dev/null 2>&1; then
  url="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VER}/protoc-${PROTOC_VER}-linux-x86_64.zip"
  curl -fsSL "$url" -o "$TMPDIR/protoc.zip"
  python3 -m zipfile -e "$TMPDIR/protoc.zip" "$HOME/.local/"
  chmod +x "$HOME/.local/bin/protoc"
fi

uv venv "$ENV_DIR" --python 3.11
# Base install only. sglang[all] pulls a Rust grammar lib (xgrammar) unneeded for plain
# generation. MAX_JOBS caps nvcc parallelism to bound the CUDA-extension build's memory.
MAX_JOBS="${MAX_JOBS:-4}" PROTOC="$HOME/.local/bin/protoc" VIRTUAL_ENV="$ENV_DIR" \
  uv pip install "git+https://github.com/sgl-project/sglang.git@${SGLANG_REF}#subdirectory=python"

"$ENV_DIR/bin/python" -c 'import sglang, torch; print("sglang", sglang.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
