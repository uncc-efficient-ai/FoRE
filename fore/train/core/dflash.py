# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fore.train.modeling.draft.dflash import (
    DFlashDraftModel,
)

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


_VALID_LOSS_TYPES = {
    "dpace",
}

# FORE-VARIABLE-PREFIX: the forced-prefix length distribution used for the FoRE drafters.
VARIABLE_PREFIX_MARGINAL_P = 0.5  # P(L=0): half the blocks stay pure-marginal
VARIABLE_PREFIX_L1_FRAC = 0.35  # of the forced half, the share that forces exactly slot 1
VARIABLE_PREFIX_TAIL_RATIO = 0.8  # truncated-geometric ratio for L=2..block_size-1


def variable_prefix_probs(block_size: int = 16) -> list:
    """Per-length probabilities P(L) over L=0..block_size-1 for variable-prefix forcing.

    L=0 keeps the block marginal; the forced half splits into a fixed slot-1 share and a
    truncated-geometric tail over L=2..block_size-1.
    """
    forced = 1.0 - VARIABLE_PREFIX_MARGINAL_P
    probs = [0.0] * block_size
    probs[0] = VARIABLE_PREFIX_MARGINAL_P
    probs[1] = forced * VARIABLE_PREFIX_L1_FRAC
    tail = [VARIABLE_PREFIX_TAIL_RATIO**k for k in range(block_size - 2)]
    tail_mass = forced * (1.0 - VARIABLE_PREFIX_L1_FRAC)
    for k, w in enumerate(tail):
        probs[2 + k] = tail_mass * w / sum(tail)
    return probs


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional; different blocks are invisible.
      3. Invalid blocks (block_keep_mask=False) see nothing.
    """
    B, N = anchor_positions.shape

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = (kv_idx < S) & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with the D-PACE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_type: str = "dpace",
        dpace_alpha: float = 0.5,
        cond_prefix_probs: Optional[
            list
        ] = None,  # FORE-VARIABLE-PREFIX: per-length P(L) forcing dist
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha

        # FORE-VARIABLE-PREFIX: variable-length true-prefix forcing. cond_prefix_probs[L] is P(L)
        # over L=0..block_size-1; None disables forcing. Stored
        # as plain lists (not buffers) because the wrapper is cast to bf16 in training and a
        # bf16 CDF would collapse the length bins; _prefix_length rebuilds an fp CDF per call.
        self.cond_prefix_probs = None
        self._cond_prefix_order = None
        self._cond_prefix_cdf = None
        if cond_prefix_probs is not None:
            if len(cond_prefix_probs) != block_size:
                raise ValueError(
                    f"cond_prefix_probs must have length block_size={block_size}, "
                    f"got {len(cond_prefix_probs)}"
                )
            if any(p < 0 for p in cond_prefix_probs):
                raise ValueError("cond_prefix_probs entries must be non-negative")
            total = float(sum(cond_prefix_probs))
            if total <= 0:
                raise ValueError("cond_prefix_probs must sum to a positive value")
            probs = [p / total for p in cond_prefix_probs]
            # Order L as [1, 2, ..., block_size-1, 0] so forcing (L>=1) occupies the low-u
            # region; u is uniform, so the marginal over L is unchanged by the ordering.
            order = list(range(1, block_size)) + [0]
            cdf = []
            acc = 0.0
            for L in order:
                acc += probs[L]
                cdf.append(acc)
            cdf[-1] = 1.0
            self.cond_prefix_probs = probs
            self._cond_prefix_order = order
            self._cond_prefix_cdf = cdf

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(
        self,
        input_ids,
        anchor_positions,
        block_keep_mask,
    ):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        # FORE-VARIABLE-PREFIX: teacher-force a true-token prefix into each block so the
        # drafter learns p(suffix | forced prefix), the distribution it drafts from when a
        # tree supplies the prefix. The forced length, sampled from cond_prefix_probs, is a
        # variable length L>=0 (slots 1..L). Without cond_prefix_probs no prefix is forced and
        # no RNG is drawn.
        forced = torch.zeros(bsz, n, bs, dtype=torch.bool, device=device)
        if self.cond_prefix_probs is not None:
            slot = torch.arange(bs, device=device).view(1, 1, bs)
            tok_positions = anchor_positions.unsqueeze(-1) + slot
            within_bounds = tok_positions < seq_len
            true_tokens = torch.gather(
                input_ids.unsqueeze(1).expand(-1, n, -1),
                2,
                tok_positions.clamp(0, seq_len - 1),
            )
            keep = block_keep_mask.unsqueeze(-1)
            lengths = self._prefix_length(torch.rand(bsz, n, device=device))
            forced = (
                (slot >= 1) & (slot <= lengths.unsqueeze(-1)) & keep & within_bounds
            )
            noise_ids = torch.where(
                forced.reshape(bsz, n * bs), true_tokens.reshape(bsz, n * bs), noise_ids
            )

        return self.embed_tokens(noise_ids), forced

    def _prefix_length(self, u: torch.Tensor) -> torch.Tensor:
        """Map one uniform per block to a forced-prefix length via the variable-prefix CDF.

        The CDF is laid out over L ordered [1, 2, ..., block_size-1, 0], so forcing lives in
        the low-u region."""
        cdf = torch.tensor(self._cond_prefix_cdf, dtype=u.dtype, device=u.device)
        order = torch.tensor(self._cond_prefix_order, dtype=torch.long, device=u.device)
        idx = torch.searchsorted(cdf, u.contiguous(), right=True).clamp(
            max=self.block_size - 1
        )
        return order[idx]

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        return torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding, forced_mask = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        # KV order is [context | draft]; position ids must match so RoPE lines up.
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        # FORE-VARIABLE-PREFIX: forced-prefix slots carry the answer in their input; their CE is an
        # echo task, so they get zero loss weight.
        weight_mask = weight_mask * (1.0 - forced_mask.float())

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Per-token loss: cross-entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        metrics: dict = {}
        with torch.no_grad():
            q = torch.exp(-ce_per_token).view_as(target_ids)
            dpace_weights = self._dpace_weight(q, weight_mask, weight_mask > 0)
        loss_weights = weight_mask * dpace_weights
        loss = (ce_per_token.view_as(target_ids) * loss_weights).sum() / float(bsz)

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy, metrics
