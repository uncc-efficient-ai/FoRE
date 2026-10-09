"""Preallocated K/V cache for draft rows that share one growing context.

`update` writes at [ctx_len : ctx_len + n] and returns views up to that point without
advancing ctx_len, so a draft pass's block positions are scratch that the next pass
overwrites. `commit(n)` advances ctx_len once the committed context has been written. A
1-row write broadcasts to every row, since the context is identical across rows. This
avoids re-copying the expanded context on every pass, which DynamicCache's concatenation
does.
"""


class StaticRowsCache:
    def __init__(self, rows, max_len):
        self.rows = rows
        self.max_len = max_len
        self.ctx_len = 0
        self._k = {}
        self._v = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self._k:
            shape = (self.rows, key_states.shape[1], self.max_len, key_states.shape[3])
            self._k[layer_idx] = key_states.new_empty(shape)
            self._v[layer_idx] = value_states.new_empty(shape)
        n_rows, _, n, _ = key_states.shape
        end = self.ctx_len + n
        if end > self.max_len or n_rows > self.rows:
            raise ValueError(f"cache overflow: {n_rows} rows to {end} positions")
        rows = slice(None) if n_rows == 1 else slice(0, n_rows)
        self._k[layer_idx][rows, :, self.ctx_len : end] = key_states
        self._v[layer_idx][rows, :, self.ctx_len : end] = value_states
        return self._k[layer_idx][:n_rows, :, :end], self._v[layer_idx][:n_rows, :, :end]

    def commit(self, n):
        self.ctx_len += n

    def get_seq_length(self, layer_idx=0):
        return self.ctx_len

