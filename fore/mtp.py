"""The target's own MTP module, rebuilt from its checkpoint.

transformers loads Qwen3.5 without its `mtp.*` tensors, so the module is reconstructed from
the safetensors on disk. The template is a copy of a full-attention decoder layer, and the
MTP weights are copied into it with coverage asserted both ways.
"""

import json
from copy import deepcopy
from pathlib import Path

import safetensors.torch as st
import torch
from torch import nn


class Qwen35MTPModule(nn.Module):
    def __init__(self, block, pre_fc_norm_embedding, pre_fc_norm_hidden, fc, norm, rotary_emb):
        super().__init__()
        self.block = block
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.fc = fc
        self.norm = norm
        self._rotary_emb = rotary_emb
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def uses_flash_attention(self):
        return self.block.self_attn.config._attn_implementation == "flash_attention_2"

    def forward(self, trunk_hidden, next_embed, position_ids, attention_mask, past_key_values):
        """The module's state at each position, given the trunk state and the next token."""
        fused = self.fc(
            torch.cat([self.pre_fc_norm_embedding(next_embed), self.pre_fc_norm_hidden(trunk_hidden)], dim=-1)
        )
        cos, sin = self._rotary_emb(fused, position_ids)
        out = self.block(
            fused,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if isinstance(out, tuple):
            out = out[0]
        return self.norm(out)


def _mtp_tensors(model_path):
    model_dir = Path(model_path)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = {f for k, f in weight_map.items() if k.startswith("mtp.")}
    else:
        shards = {"model.safetensors"}
    tensors = {}
    for shard in shards:
        with st.safe_open(str(model_dir / shard), framework="pt") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")})
    if not tensors:
        raise ValueError(f"{model_path} has no mtp.* tensors")
    return tensors


def load_mtp(target, model_path):
    text_cfg = target.config.text_config
    lm = target.model.language_model
    template = lm.layers[text_cfg.layer_types.index("full_attention")]
    param = next(template.parameters())
    device, dtype = param.device, param.dtype
    hidden, eps = text_cfg.hidden_size, text_cfg.rms_norm_eps
    rms_norm = type(template.input_layernorm)

    block = deepcopy(template).to(dtype=dtype, device=device)
    parts = {
        "pre_fc_norm_embedding": rms_norm(hidden, eps=eps),
        "pre_fc_norm_hidden": rms_norm(hidden, eps=eps),
        "fc": nn.Linear(2 * hidden, hidden, bias=False),
        "norm": rms_norm(hidden, eps=eps),
    }
    parts = {k: m.to(dtype=dtype, device=device) for k, m in parts.items()}

    tensors = _mtp_tensors(model_path)
    targets = {f"mtp.{name}.weight": module.weight for name, module in parts.items()}
    targets.update({f"mtp.layers.0.{n}": p for n, p in block.named_parameters()})
    if set(targets) != set(tensors):
        raise ValueError(
            f"MTP layout mismatch: missing {sorted(set(targets) - set(tensors))}, "
            f"unused {sorted(set(tensors) - set(targets))}"
        )
    with torch.no_grad():
        for key, p in targets.items():
            p.copy_(tensors[key].to(dtype=p.dtype, device=p.device))
    return Qwen35MTPModule(block, rotary_emb=lm.rotary_emb, **parts)
