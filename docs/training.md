# Training and reproduction

## Train a drafter

```bash
uv sync --locked --extra flash-attn --extra train
uvx --from huggingface_hub \
  hf download z-lab/Qwen3.5-9B-DFlash \
  --revision 492f4b532a957a50561e1418e5a3f31690f127f4 \
  --local-dir models/Qwen3.5-9B-DFlash
TARGET=models/Qwen3.5-9B \
DFLASH=models/Qwen3.5-9B-DFlash \
  bash scripts/train_drafter.sh 9b
```

`scripts/train_drafter.sh` runs the six steps below and prints the drafter's directory when it
finishes. A step whose output exists is skipped, so after an interruption, run the same command
again. For Qwen3.5-4B, pass `4b` with the 4B target and `z-lab/Qwen3.5-4B-DFlash` at its pinned
revision[^revisions]; it reuses steps 1-3 if the 9B ran first. `uv sync` installs exactly the
extras it is given and removes the others.

Before the first run:

- Step 1 reads Nemotron-Post-Training-Dataset-v2, which is gated on Hugging Face. Accept its
  terms, then log in or set `HF_TOKEN`.
- Step 4 runs in a separate SGLang environment, built into `./sglang-env` on first use. If Rust
  or protoc is missing, the build installs Rust with rustup into `~/.cargo` and protoc into
  `~/.local`.
- The 9B fine-tune needs four GPUs with more than 48 GB each; ours ran on four H200s.

| Variable | Default | |
|---|---|---|
| `TARGET` | required | the target model directory |
| `DFLASH` | required | the released DFlash drafter at its pinned revision[^revisions] |
| `N` | `100000` | prompts to regenerate |
| `NGPU` | `8` | GPUs for regeneration, one SGLang server each |
| `NPROC` | `4` | GPUs for fine-tuning |
| `DATA` | `data` | directory for the data files |
| `RUNS` | `runs` | directory for the checkpoints |
| `ENV_DIR` | `sglang-env` | the SGLang environment |

Relative paths are taken from the directory you run the script in; the defaults are inside the
repository.

| Step | Script | Writes |
|---|---|---|
| 1. Prompt pool | `scripts/prep_prompts.py` | `data/prompts.parquet` |
| 2. Length filter | `scripts/run_gen.sh` | `data/pool/completions.parquet` |
| 3. 100K sample | `scripts/build_regen_input.py` | `data/regen-100000.jsonl` |
| 4. Responses | `scripts/run_regen.sh` | `data/regen-100000-qwen3.5-9b-out.jsonl` |
| 5. Split | `scripts/data_prep.py` | `data/train-100000-qwen3.5-9b.jsonl` |
| 6. Fine-tune | `fore.train.train_dflash` | `runs/fore-qwen3.5-9b-100000/epoch_1_step_<N>/` |

The drafter is the last checkpoint, `epoch_1_step_<N>`[^steps]. Pass its directory to
`fore-generate --drafter` or `fore.load`. Sampling in step 4 is not deterministic, so your
responses, and the drafter trained on them, will differ from ours.

## Running one step

These are the commands `train_drafter.sh` runs for Qwen3.5-9B. For the 4B, use `qwen3.5-4b` in
the file names.

### Steps 1-3: prompts

```bash
uv run python scripts/prep_prompts.py data/prompts.parquet
PY=.venv/bin/python bash scripts/run_gen.sh \
  models/Qwen3.5-9B \
  data/prompts.parquet \
  data/pool \
  7
uv run python scripts/build_regen_input.py \
  --parquet data/pool/completions.parquet \
  --n 100000 \
  --out data/regen-100000.jsonl
```

Keep the shard count at 7. Deduplication keeps the first copy in shard order, so another count
selects different prompts[^pool].

### Step 4: responses

```bash
bash scripts/build_sglang_env.sh
WAIT=1 \
MODEL=models/Qwen3.5-9B \
NGPU=8 \
INPUT=data/regen-100000.jsonl \
OUTPUT=data/regen-100000-qwen3.5-9b-out.jsonl \
  bash scripts/run_regen.sh
```

`run_regen.sh` starts one SGLang server per GPU, on the GPUs in `CUDA_VISIBLE_DEVICES` when it
is set. With `WAIT=1` it waits for the client and stops the servers when the client exits or the
script is interrupted. Without it, the script returns with the client running in the background;
stop the servers afterwards with `kill` and the PIDs it printed (`server gpu=... pid=...`). The
client exits with an error if any prompt failed, listing them in
`regen-100000-qwen3.5-9b-out_error.jsonl`; a rerun skips finished prompts and retries the
failed ones. The SGLang environment is not covered by `uv.lock`[^sglang].

To use a server you already run, install the client with `uv sync --extra regen` and call
`scripts/regenerate_train_data.py` with these sampling flags:

```bash
--no-thinking
--temperature 0.7
--top-p 0.95
--max-tokens 3072
```

### Step 5: split

`data/` holds the prompt ids we held out for each target[^holdout], and
`decontam-exclude-ids.txt` the 107 prompts excluded from the 9B training set for overlap with an
evaluation set[^decontam]. The 4B excludes only its held-out ids,
`data/holdout-ids-qwen3.5-4b.txt`.

```bash
cat data/decontam-exclude-ids.txt data/holdout-ids-qwen3.5-9b.txt > data/exclude-qwen3.5-9b.txt
uv run python scripts/data_prep.py \
  data/regen-100000-qwen3.5-9b-out.jsonl \
  data/train-100000-qwen3.5-9b.jsonl \
  --exclude-ids data/exclude-qwen3.5-9b.txt
```

The training row counts on our responses are in the footnote[^rows].

### Step 6: fine-tune

```bash
uv run torchrun \
  --standalone \
  --nproc-per-node 4 \
  -m fore.train.train_dflash \
  --target-model-path models/Qwen3.5-9B \
  --target-model-backend hf \
  --init-from models/Qwen3.5-9B-DFlash \
  --train-data-path data/train-100000-qwen3.5-9b.jsonl \
  --block-size 16 \
  --max-length 3072 \
  --num-anchors 512 \
  --chat-template qwen3-nosys \
  --embedding-key model.language_model.embed_tokens.weight \
  --trust-remote-code \
  --num-epochs 1 \
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
  --output-dir runs/fore-qwen3.5-9b-100000 \
  --cache-dir data/cache
```

`--init-from` must be the released DFlash drafter at its pinned revision[^revisions]; later
revisions have a different shape. `--resume` continues from the last checkpoint in
`--output-dir`, using the optimizer state in its `training_state.pt`, or starts from
`--init-from` when there is none. The tokenized-data cache in `--cache-dir` is keyed by the
train file's path, so clear it after rewriting the train file; `train_drafter.sh` does. The
trainer's other settings are its defaults[^trainer].

## `fore-generate` options

| Flag | |
|---|---|
| `--target` | a local copy of `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-4B`; the MTP weights are read from its safetensors |
| `--drafter` | the FoRE drafter for the same target |
| `--k` | the number K of MTP-guided drafts |
| `--r` | the replacement budget R, the number of additional drafts from draft refinement (0 for MTP-guided early forking alone) |
| `--max-new-tokens` | generation stops at the end of the step that crosses it, so it can run up to one block past it |
| `--attention` | FlashAttention 2 by default; `sdpa` without the `flash-attn` extra (`attention="sdpa"` for `fore.load`) |

## Environment

`uv.lock` pins the environment. In it, this package produces the same tokens and per-step
acceptance as our research code; a different GPU, build or Python version can flip bf16
near-ties.

---

[^revisions]: `z-lab/Qwen3.5-9B-DFlash` at `492f4b532a957a50561e1418e5a3f31690f127f4` and
    `z-lab/Qwen3.5-4B-DFlash` at `96899cc270945f554998309580b08a04a05a3187`, as listed in
    [THIRD_PARTY.md](../THIRD_PARTY.md).
[^steps]: N = 24,046 on our 9B data and 23,926 on our 4B data.
[^pool]: Steps 2 and 3 reproduce our `regen-100000.jsonl` from our step-1 pool.
[^sglang]: Our responses used SGLang `53ed34cb8`, torch 2.11.0, flashinfer 0.6.12, transformers
    5.8.1 and Python 3.11.
[^holdout]: The held-out sets are `--holdout-n 2000 --holdout-seed 7` samples of the
    regeneration output in file order. The 9B set excludes the three decontaminated prompts it
    contained.
[^decontam]: A prompt is excluded when its regenerated row shares at least eight distinct word
    8-grams, or at least 30% of a test item's 8-grams, with a GSM8K, MATH-500, HumanEval, MBPP or
    MT-Bench test item.
[^rows]: 97,896 training rows for the 9B and 98,000 for the 4B.
[^trainer]: AdamW (weight decay 0) on fp32 master weights, gradient-norm clipping at 1.0, linear
    warmup over the first 4% of steps, then cosine decay. The command overrides the upstream
    defaults of 6 epochs and learning rate 6e-4.
