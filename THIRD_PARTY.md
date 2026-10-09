# Third-party code, models and data

## Code

| Where | Source | License |
|---|---|---|
| `fore/train/` | [D-PACE](https://github.com/Lucas-TY/D-PACE) at `f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f`, a fork of [SpecForge](https://github.com/sgl-project/SpecForge) that adds the D-PACE loss. Files keep their upstream names and layout. Most code the fine-tuning recipe does not use is deleted; our main additions (initialization from a released drafter, visible prefixes of varying lengths, multimodal target loading) are marked `FORE-`. | MIT, (c) 2025 sgl-project, the license of both D-PACE and SpecForge: [`licenses/SpecForge-MIT.txt`](licenses/SpecForge-MIT.txt) |
| `fore/train/data/preprocessing.py`, `fore/train/data/utils.py` | As above; these files carry code from EleutherAI GPT-NeoX, Hugging Face Transformers and [EAGLE](https://github.com/SafeAILab/EAGLE), under the Apache-2.0 headers kept at the top of each file. Modified: code for paths the recipe does not use is removed. | Apache-2.0: [`LICENSE`](LICENSE) |
| `fore/train/data/template.py` | As above; adapted from [SGLang](https://github.com/sgl-project/sglang)'s `chat_template.py`. Modified: only the template the recipe uses is kept. | Apache-2.0: [`LICENSE`](LICENSE) |
| `scripts/regenerate_train_data.py` | Adapted from SpecForge's `scripts/regenerate_train_data.py` (resume keyed by id). | MIT, (c) 2025 sgl-project |
| `fore/drafter.py` | The DFlash drafter architecture of SpecForge's `specforge/modeling/draft/dflash.py`, itself a port of [DFlash](https://github.com/z-lab/dflash). | MIT, (c) 2025 sgl-project; MIT, (c) 2026 Z Lab: [`licenses/dflash-MIT.txt`](licenses/dflash-MIT.txt) |
| `fore/rows.py` | Adapted from [BASTION](https://github.com/kaist-ai-osi-lab/BASTION) (arXiv 2605.29727). | MIT, (c) 2026 Z Lab, as in BASTION's [LICENSE](https://github.com/kaist-ai-osi-lab/BASTION/blob/main/LICENSE); same text as [`licenses/dflash-MIT.txt`](licenses/dflash-MIT.txt) |

## Models

The targets `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` are Apache-2.0.

The FoRE drafters are fine-tuned from the released DFlash drafters, both MIT:
`z-lab/Qwen3.5-9B-DFlash` at revision `492f4b532a957a50561e1418e5a3f31690f127f4` and
`z-lab/Qwen3.5-4B-DFlash` at revision `96899cc270945f554998309580b08a04a05a3187`. Later
revisions of both repositories hold a retrained, differently shaped drafter.

## Data

Training prompts come from `nvidia/Nemotron-Post-Training-Dataset-v2` (revision
`5c89e01dd720ae0f4058445ed49c5fb68a03c76e`) and `sahil2801/CodeAlpaca-20k` (revision
`152bb5e9a29651266b018106053980070a0521a1`). The Nemotron dataset card labels
the collection CC BY 4.0 but lists per-sample license exceptions. The CodeAlpaca
[Hugging Face card](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) labels it CC BY 4.0,
while the [upstream data license](https://github.com/sahil280114/codealpaca/blob/master/DATA_LICENSE)
says CC BY-NC 4.0. `data/` holds only prompt ids into the pool that
`scripts/prep_prompts.py` builds from them.
