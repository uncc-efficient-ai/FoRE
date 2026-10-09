# FoRE: Fork and Regenerate

Code for *Fork and Regenerate: MTP-Guided Early Forking and Suffix Regeneration for Block
Diffusion Speculative Decoding* (Bruh Lemma Yadecha and Li Yang).

Models: [Hugging Face collection](https://huggingface.co/collections/byadecha/fore-6ac86bb87b4d14bc6f429f9d)

Inference and fine-tuning code for FoRE on Qwen3.5-9B and Qwen3.5-4B, with a fine-tuned DFlash
block diffusion drafter. The output is the target's greedy continuation, up to bf16 near-ties.

Each decoding step:

1. MTP-guided early forking: the target's pretrained MTP module proposes the top-K candidates
   for the first drafted position. One batched drafter pass generates the rest of a block for
   each candidate, with the candidate supplied as the block's first draft token.
2. Draft refinement: replacement tokens at later positions are ranked by the probabilities of
   their retained prefix and the replacement, reusing the distributions from step 1. The top R
   become additional drafts, and a second drafter pass regenerates the suffix after each
   replacement.
3. The target verifies all K+R drafts as rows of one batched forward pass and commits the longest
   prefix that matches its greedy predictions, plus its own next token.

## Setup

Requires Linux, an NVIDIA GPU, CUDA 13.0 (`nvcc` on `PATH` to build `causal-conv1d`) and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked                                   # inference, sdpa attention
uv sync --locked --extra flash-attn                # plus FlashAttention 2, the backend used for timing
uv sync --locked --extra flash-attn --extra train  # plus fine-tuning and the training-data scripts
```

`uv sync` installs exactly the extras it is given and removes the others.

`uv.lock` pins the training and inference environment. In it, this package produces the same
tokens and per-step acceptance as our research code; a different GPU, build or Python version
can flip bf16 near-ties. The SGLang server for response generation is installed separately
and is not covered by `uv.lock` (see [Responses](#responses)).

## Run

Download a target and its FoRE drafter, `byadecha/Qwen3.5-9B-FoRE` or `byadecha/Qwen3.5-4B-FoRE`:

```bash
uvx --from huggingface_hub hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
uvx --from huggingface_hub hf download byadecha/Qwen3.5-9B-FoRE --local-dir models/Qwen3.5-9B-FoRE
```

```bash
uv run fore-generate \
  --target models/Qwen3.5-9B \
  --drafter models/Qwen3.5-9B-FoRE \
  --k 8 --rows 8 \
  --prompt "Write a Python function that returns the n-th Fibonacci number."
```

`--target` is a local copy of `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.5-4B`; the MTP weights are read
from its safetensors. `--drafter` is the FoRE drafter for the same target. `--k` is the number
K of MTP-guided drafts, and `--rows` the replacement budget R, the number of additional drafts
from draft refinement (0 for MTP-guided early forking alone). The paper's settings are
`--k 8 --rows 0` and `--k 8 --rows 8`. Generation stops at the end of the step that crosses
`--max-new-tokens`, so it can run up to one block past it.
The default attention is FlashAttention 2; without the `flash-attn` extra, pass
`--attention sdpa` here and `attention="sdpa"` to `load` below.

From Python:

```python
from fore import load, generate

target, drafter, mtp, tokenizer = load("models/Qwen3.5-9B", "models/Qwen3.5-9B-FoRE")
ids, steps = generate(target, drafter, mtp, input_ids, max_new_tokens=512,
                      stop_ids=[tokenizer.eos_token_id], k=8, rows=8)
```

`steps` holds the tokens committed per step; its mean is the acceptance length τ.

## Training

Training fine-tunes a released DFlash drafter for one epoch on responses the target itself
generated, with visible prefixes of varying lengths in each block and the D-PACE loss. The
same prompt set serves both targets; each target regenerates its own responses.

### Prompts

```bash
# 1. Pool: 150K chat, 50K code, 30K math prompts from Nemotron-Post-Training-Dataset-v2 and all
#    of CodeAlpaca-20k, at pinned revisions (the Nemotron dataset is gated on Hugging Face).
uv run python scripts/prep_prompts.py data/prompts.parquet
# 2. Keep prompts whose chat-templated form is at most 1,024 tokens. Deduplication keeps the
#    first copy in shard order, so the shard count must stay 7.
PY=.venv/bin/python bash scripts/run_gen.sh /path/to/Qwen3.5-9B data/prompts.parquet data/pool 7
# 3. Deduplicate and take the first 100,000 prompts of a source-stratified order.
uv run python scripts/build_regen_input.py --parquet data/pool/completions.parquet \
  --n 100000 --out data/regen-100000.jsonl
```

Steps 2 and 3 reproduce our `regen-100000.jsonl` from our step-1 pool.

### Responses

The target regenerates each response with SGLang at temperature 0.7, top-p 0.95, thinking
off, and at most 3,072 new tokens. SGLang runs in its own environment. `scripts/run_regen.sh`
starts one server per GPU and drives them with `scripts/regenerate_train_data.py`; a rerun
resumes by prompt id. Our responses used SGLang `53ed34cb8`, torch 2.11.0, flashinfer 0.6.12,
transformers 5.8.1 and Python 3.11. Sampling is not deterministic, so regenerated responses
can differ from ours.

```bash
bash scripts/build_sglang_env.sh     # into ./sglang-env
MODEL=/path/to/Qwen3.5-9B NGPU=8 INPUT=data/regen-100000.jsonl \
  OUTPUT=data/regen-100000-qwen3.5-9b-out.jsonl bash scripts/run_regen.sh
```

`run_regen.sh` returns with the client running in the background. When the client exits, stop
the servers with `kill` and the PIDs `run_regen.sh` printed (`server gpu=... pid=...`).

For the 4B, set `MODEL` to Qwen3.5-4B and a separate `OUTPUT`. For an already running server,
install the client with `uv sync --extra regen` and call
`scripts/regenerate_train_data.py` with `--no-thinking --temperature 0.7 --top-p 0.95
--max-tokens 3072`.

### Train/held-out split

`data/` lists the prompt ids behind our splits. `decontam-exclude-ids.txt` holds the 107
prompts whose regenerated row shares at least eight distinct word 8-grams, or at least 30% of a
test item's 8-grams, with a GSM8K, MATH-500, HumanEval, MBPP or MT-Bench test item.

```bash
# Qwen3.5-9B: 1,997 held-out prompts, decontaminated -> 97,896 training rows
cat data/decontam-exclude-ids.txt data/holdout-ids-qwen3.5-9b.txt > exclude-9b.txt
uv run python scripts/data_prep.py data/regen-100000-qwen3.5-9b-out.jsonl \
  data/train-3072-qwen3.5-9b.jsonl --exclude-ids exclude-9b.txt
# Qwen3.5-4B: 2,000 held-out prompts, no decontamination -> 98,000 training rows
uv run python scripts/data_prep.py data/regen-100000-qwen3.5-4b-out.jsonl \
  data/train-3072-qwen3.5-4b.jsonl --exclude-ids data/holdout-ids-qwen3.5-4b.txt
```

The held-out sets are `--holdout-n 2000 --holdout-seed 7` samples of the regeneration output
in file order. The 9B set excludes the three decontaminated prompts it contained.

### Fine-tuning

```bash
uv run torchrun --standalone --nproc-per-node 4 -m fore.train.train_dflash \
  --target-model-path /path/to/Qwen3.5-9B --target-model-backend hf \
  --init-from /path/to/Qwen3.5-9B-DFlash \
  --train-data-path data/train-3072-qwen3.5-9b.jsonl \
  --block-size 16 --max-length 3072 --num-anchors 512 \
  --chat-template qwen3-nosys --embedding-key model.language_model.embed_tokens.weight \
  --trust-remote-code --num-epochs 1 --batch-size 1 --learning-rate 1e-4 \
  --loss-type dpace --dpace-alpha 0.5 --variable-prefix \
  --attention-backend flex_attention --seed 42 \
  --log-interval 20 --save-interval 1000 \
  --output-dir runs/fore-qwen3.5-9b --cache-dir cache
```

For the 4B, change the target, `--init-from`, `--train-data-path` and `--output-dir`; all other
flags are the same. `--init-from` must be the released drafter at the revision in
[THIRD_PARTY.md](THIRD_PARTY.md); later revisions have a different shape. Trainer defaults:
AdamW (weight decay 0) on fp32 master weights, gradient-norm clipping at 1.0, linear warmup over
the first 4% of steps, then cosine decay. The command overrides the upstream defaults of 6
epochs and learning rate 6e-4.

The FoRE drafter is the last checkpoint, `epoch_1_step_<N>` (N = 24,046 on our 9B data and
23,926 on our 4B data). Its `config.json` and `model.safetensors` load with
`fore-generate --drafter` and `fore.load`;
`training_state.pt` is the optimizer state for `--resume`. The 9B fine-tune ran on four H200s and
needs more than 48 GB per GPU at 512 anchors and 3,072 tokens.

## Layout

| File | Contents |
|---|---|
| `fore/decode.py` | model loading and the decoding loop |
| `fore/drafter.py` | the DFlash drafter architecture |
| `fore/mtp.py` | the target's MTP module, rebuilt from its checkpoint |
| `fore/rows.py` | replacement selection for draft refinement |
| `fore/verify.py` | batched verification on the hybrid target |
| `fore/cache.py` | the preallocated drafter cache |
| `fore/train/` | the fine-tuning trainer: `train_dflash.py`, the objective in `core/dflash.py` |
| `scripts/` | prompt pool, regeneration and split |
| `data/` | prompt ids of the held-out sets and the decontamination list |

## License

FoRE is released under the Apache License 2.0 ([LICENSE](LICENSE)). Third-party code, models
and data keep their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md).
