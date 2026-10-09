"""Batched verification of draft rows on a hybrid (attention + GatedDeltaNet) target.

All rows share one committed prefix. Before each verify, the recurrent states are restored
from a snapshot of that prefix and the attention cache is broadcast across the rows. After
it, the winning row is committed without a second target forward. Attention K/V is sliced
from the verify pass. Each GatedDeltaNet state is advanced by replaying only the recurrent
kernel over the winner's accepted tokens, from inputs stashed during the verify.
"""

import torch


class HybridTreeVerifier:
    def __init__(self, target):
        base = target.model.language_model
        layers = base.layers
        self.target = target
        self.gdn = {
            i: layer.linear_attn
            for i, layer in enumerate(layers)
            if getattr(layer, "layer_type", None) == "linear_attention"
        }
        self.conv_kernel = next(iter(self.gdn.values())).conv_kernel_size
        self._snap = {}
        self._prefix_len = 0
        self._conv_ws = {}
        self._rec_ws = {}
        self._capturing = False
        self._cap = {}
        # One verifier is built per generate() call. Replace any earlier verifier's hook
        # and wrapper, since stacking them keeps every dead verifier's stash alive.
        for i, mix in self.gdn.items():
            prev = getattr(mix.in_proj_qkv, "_hv_handle", None)
            if prev is not None:
                prev.remove()
            mix.in_proj_qkv._hv_handle = mix.in_proj_qkv.register_forward_hook(self._inproj_hook(i))
            orig = getattr(mix, "_hv_orig_chunk", None) or mix.chunk_gated_delta_rule
            mix._hv_orig_chunk = orig
            mix.chunk_gated_delta_rule = self._wrap_chunk(i, orig)

    def _inproj_hook(self, i):
        def hook(_module, _inp, out):
            if self._capturing:
                self._cap.setdefault(i, {})["mixed_qkv"] = out.detach()

        return hook

    def _wrap_chunk(self, i, orig):
        def wrapped(query, key, value, g, beta, **kw):
            out, final = orig(query, key, value, g=g, beta=beta, **kw)
            if self._capturing:
                self._cap.setdefault(i, {}).update(
                    q=query.detach(), k=key.detach(), v=value.detach(), g=g.detach(), beta=beta.detach()
                )
            return out, final

        return wrapped

    def capture_prefix_snapshot(self, prefix_len, past_key_values):
        self._prefix_len = prefix_len
        self._snap = {
            i: (layer.conv_states.clone(), layer.recurrent_states.clone())
            for i, layer in enumerate(past_key_values.layers)
            if hasattr(layer, "recurrent_states")
        }
        assert self._snap, "no recurrent cache layers found; transformers' cache layout has changed"

    def _prepare(self, past_key_values, rows):
        for i, layer in enumerate(past_key_values.layers):
            if getattr(layer, "keys", None) is not None:
                if layer.keys.shape[0] != rows:
                    if rows == 1:
                        layer.keys, layer.values = layer.keys[0:1], layer.values[0:1]
                    else:
                        layer.keys = layer.keys.expand(rows, -1, -1, -1)
                        layer.values = layer.values.expand(rows, -1, -1, -1)
            if i in self._snap:
                conv, rec = self._snap[i]
                if rows == 1:
                    layer.conv_states, layer.recurrent_states = conv.clone(), rec.clone()
                    continue
                for ws, snap, name in ((self._conv_ws, conv, "conv_states"), (self._rec_ws, rec, "recurrent_states")):
                    if i not in ws or ws[i].shape[0] < rows:
                        ws[i] = snap.new_empty((max(rows, 8),) + snap.shape[1:])
                    setattr(layer, name, ws[i][:rows].copy_(snap))
        return past_key_values

    def verify(self, rows, position_ids, past_key_values):
        """Logits and hidden states for every row, stashing the kernel inputs for commit()."""
        kv = self._prepare(past_key_values, rows.shape[0])
        self._cap = {}
        self._capturing = True
        try:
            out = self.target(
                input_ids=rows,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=kv,
                use_cache=True,
                output_hidden_states=True,
            )
        finally:
            self._capturing = False
        return out.logits, out.hidden_states

    def commit(self, w, kept, past_key_values):
        """Advance the cache to the prefix plus row w's first `kept` tokens."""
        past_key_values.crop(self._prefix_len + kept)
        for i, layer in enumerate(past_key_values.layers):
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[w : w + 1].contiguous()
                layer.values = layer.values[w : w + 1].contiguous()
                continue
            conv0, rec0 = self._snap[i]
            cap = self._cap[i]
            # The recurrent kernel, not the chunked one: over a few tokens the chunked
            # kernel pads to 64 and costs several times more.
            _, state = self.gdn[i].recurrent_gated_delta_rule(
                cap["q"][w : w + 1, :kept],
                cap["k"][w : w + 1, :kept],
                cap["v"][w : w + 1, :kept],
                g=cap["g"][w : w + 1, :kept],
                beta=cap["beta"][w : w + 1, :kept],
                initial_state=rec0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            layer.recurrent_states = state
            conv_in = torch.cat([conv0, cap["mixed_qkv"][w : w + 1, :kept].transpose(1, 2)], dim=-1)
            layer.conv_states = conv_in[..., -self.conv_kernel :].contiguous()
        self._cap = {}
        return past_key_values
