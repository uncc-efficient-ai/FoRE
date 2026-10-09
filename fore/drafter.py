"""The DFlash block drafter, as the released checkpoints and our fine-tuned drafters store it.

Parameter names match the checkpoints exactly, so `DFlashDraftModel.from_pretrained` loads
both the released drafter and a FoRE drafter fine-tuned from it.
"""

from collections.abc import Callable

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Keys span [context | block] and queries only the block, so queries take the
    # trailing slice of the rotary table.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def extract_context_feature(hidden_states, layer_ids):
    # hidden_states[0] is the embedding output, so layer i is at index i + 1.
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        kv_dim = config.num_key_value_heads * self.head_dim
        q_dim = config.num_attention_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, past_key_values):
        bsz, q_len = hidden_states.shape[:-1]
        kv_len = target_hidden.shape[1] + q_len
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1)
        v = torch.cat([self.v_proj(target_hidden), self.v_proj(hidden_states)], dim=1)
        k = self.k_norm(k.view(bsz, kv_len, -1, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = past_key_values.update(k, v, self.layer_idx)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, _ = attn_fn(self, q, k, v, None, dropout=0.0, scaling=self.scaling)
        return self.o_proj(out.reshape(bsz, q_len, -1))


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, past_key_values):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), target_hidden, position_embeddings, past_key_values
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        dflash_config = config.dflash_config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = dflash_config["target_layer_ids"]
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config["mask_token_id"]
        self.post_init()

    def forward(self, noise_embedding, target_hidden, position_ids, past_key_values):
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, past_key_values)
        return self.norm(hidden_states)

    def append_context(self, cache, target_hidden, position_ids):
        """Write the context keys and values for newly committed tokens into `cache`.

        This is exactly what a full draft pass followed by dropping the block positions
        leaves in the cache, computed without the block. It relies on a static RoPE
        family, where each position's rotation is independent of the others.
        """
        th = self.hidden_norm(self.fc(target_hidden))
        bsz, ctx_len = th.shape[:2]
        cos, sin = self.rotary_emb(th, position_ids)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k = attn.k_norm(attn.k_proj(th).view(bsz, ctx_len, -1, attn.head_dim)).transpose(1, 2)
            v = attn.v_proj(th).view(bsz, ctx_len, -1, attn.head_dim).transpose(1, 2)
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
            cache.update(k, v, i)
