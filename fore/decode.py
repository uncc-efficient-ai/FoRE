"""Lossless greedy speculative decoding with FoRE: MTP-guided early forking and draft refinement.

Each step:
1. MTP-guided early forking: the target's MTP module proposes the top-K candidates for the
   first drafted position (t2), and one K-row drafter pass drafts a block per candidate, with
   that candidate placed at slot 1.
2. Draft refinement: best-first search over the drafter's per-slot probabilities picks up to
   `r` replacements. Each additional draft copies a draft up to some slot, swaps in a
   replacement token there, and a second drafter pass regenerates the rest of the block.
3. The target verifies all K + r drafts as rows of one batch. The draft matching the
   target's greedy continuation longest is committed, followed by the target's own next token.

The committed text is the target's greedy output, up to bf16 near-ties.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .cache import StaticRowsCache
from .drafter import DFlashDraftModel, extract_context_feature
from .mtp import load_mtp
from .rows import best_first_rows
from .verify import HybridTreeVerifier

SIBLINGS_PER_SLOT = 8


def load(target_path, drafter_path, attention="flash_attention_2", device="cuda:0"):
    target = AutoModelForImageTextToText.from_pretrained(
        target_path, dtype=torch.bfloat16, attn_implementation=attention
    ).to(device).eval()
    # Text-only decoding: drop the vision tower to free its memory.
    target.model.visual = None
    torch.cuda.empty_cache()
    drafter = DFlashDraftModel.from_pretrained(
        drafter_path, attn_implementation=attention, dtype=torch.bfloat16
    ).to(device).eval()
    rope = (drafter.config.rope_parameters or {}).get("rope_type", "default")
    if rope not in ("default", "linear", "yarn"):
        raise ValueError(f"append_context needs a static RoPE family, got {rope!r}")
    mtp = load_mtp(target, target_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    return target, drafter, mtp, tokenizer


def _mtp_mask(prefix, lo, start):
    """Causal mask for queries lo..start-1 over keys 0..start-1."""
    q_len = start - lo
    mask = torch.full((q_len, start), torch.finfo(prefix.dtype).min, dtype=prefix.dtype, device=prefix.device)
    allowed = torch.arange(start, device=prefix.device).unsqueeze(0) <= (
        lo + torch.arange(q_len, device=prefix.device)
    ).unsqueeze(1)
    return mask.masked_fill(allowed, 0.0).view(1, 1, q_len, start)


def _continuation_rows(block_ids, branches, siblings):
    rows = block_ids.expand(len(siblings), -1).clone()
    slots = []
    for j, (b, i, token) in enumerate(siblings):
        slot = i + 2  # extension slot 0 is block slot 2 (slot 0 is t1, slot 1 is t2)
        rows[j, 1:slot] = branches[b, 1:slot]
        rows[j, slot] = token
        slots.append(slot)
    return rows, slots


@torch.inference_mode()
def generate(target, drafter, mtp, input_ids, max_new_tokens, stop_ids, k=8, r=8):
    """Return (generated token ids [1, n], tokens committed per step)."""
    embed = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    block = drafter.block_size
    device = input_ids.device
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    output_ids = torch.full((1, max_len + block), drafter.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    draft_cache = StaticRowsCache(rows=max(k, r, 1), max_len=output_ids.shape[1])
    mtp_cache = StaticRowsCache(rows=1, max_len=output_ids.shape[1])
    mtp_done = 0
    verifier = HybridTreeVerifier(target)

    out = target(
        input_ids,
        position_ids=position_ids[:, :n_in],
        output_hidden_states=True,
        use_cache=True,
        logits_to_keep=1,
    )
    target_kv = out.past_key_values
    verifier.capture_prefix_snapshot(n_in, target_kv)
    output_ids[:, :n_in] = input_ids
    output_ids[:, n_in : n_in + 1] = out.logits[:, -1:].argmax(dim=-1)
    target_hidden = extract_context_feature(out.hidden_states, drafter.target_layer_ids)
    final_hidden = out.hidden_states[-1]

    stop = torch.tensor(stop_ids, device=device)
    accept_lens = []
    start = n_in
    while start < max_len:
        block_ids = output_ids[:, start : start + block].clone()
        noise = embed(block_ids)

        # MTP: feed only the positions committed since the last step; its last output is the anchor's state.
        lo = mtp_done
        prefix = final_hidden[:, lo:start, :]
        mask = None if mtp.uses_flash_attention else _mtp_mask(prefix, lo, start)
        mtp_state = mtp(
            prefix,
            embed(output_ids[:, lo + 1 : start + 1]),
            torch.arange(lo, start, device=device).unsqueeze(0),
            mask,
            past_key_values=mtp_cache,
        )[:, -1, :]
        mtp_cache.commit(start - lo)
        mtp_done = start
        root_logits = lm_head(mtp_state)[0]

        # Commit the new context to the draft cache once; both passes read it as a view.
        ctx_len = draft_cache.get_seq_length()
        drafter.append_context(draft_cache, target_hidden, position_ids[:, ctx_len:start])
        draft_cache.commit(start - ctx_len)
        draft_pos = position_ids[:, start : start + block]
        no_context = target_hidden[:, :0]

        cands = torch.topk(root_logits, k).indices
        branches = block_ids.expand(k, -1).clone()
        branches[:, 1] = cands
        branch_noise = noise.expand(k, -1, -1).clone()
        branch_noise[:, 1, :] = embed(cands)
        branch_logits = lm_head(
            drafter(
                noise_embedding=branch_noise,
                target_hidden=no_context.expand(k, -1, -1),
                position_ids=draft_pos.expand(k, -1),
                past_key_values=draft_cache,
            )[:, -block + 1 :, :]
        )
        branches[:, 2:] = branch_logits[:, 1:].argmax(dim=-1)

        verify_rows = branches
        if r:
            root_probs = torch.softmax(root_logits.float(), dim=-1)
            branch_probs = torch.softmax(branch_logits.float(), dim=-1)
            top_v, top_i = branch_probs[:, 1:].topk(SIBLINGS_PER_SLOT, dim=-1)
            siblings = best_first_rows(root_probs[cands], top_v, top_i, budget_rows=k + r)[:r]
            if siblings:
                cont, slots = _continuation_rows(block_ids, branches, siblings)
                cont_noise = noise.expand(len(siblings), -1, -1).clone()
                for j, slot in enumerate(slots):
                    cont_noise[j, 1 : slot + 1] = embed(cont[j, 1 : slot + 1])
                cont_logits = lm_head(
                    drafter(
                        noise_embedding=cont_noise,
                        target_hidden=no_context.expand(len(siblings), -1, -1),
                        position_ids=draft_pos.expand(len(siblings), -1),
                        past_key_values=draft_cache,
                    )[:, -block + 1 :, :]
                )
                for j, slot in enumerate(slots):
                    cont[j, slot + 1 :] = cont_logits[j, slot:].argmax(dim=-1)
                verify_rows = torch.cat([branches, cont], dim=0)

        n_rows = verify_rows.shape[0]
        logits, hidden = verifier.verify(
            verify_rows, position_ids[:, start : start + block].expand(n_rows, -1), target_kv
        )
        post = logits.argmax(dim=-1)
        # t2 comes from the shared prefix alone, so at most one branch survives slot 1; the
        # longest match across all rows is the committed path.
        accepted = (verify_rows[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)
        w_t = accepted.argmax()
        w, accept = torch.stack([w_t.long(), accepted[w_t].long()]).tolist()
        output_ids[:, start : start + accept + 1] = verify_rows[w : w + 1, : accept + 1]
        output_ids[:, start + accept + 1] = post[w, accept]
        kept = accept + 1
        start += kept
        target_kv = verifier.commit(w, kept, target_kv)
        verifier.capture_prefix_snapshot(start, target_kv)
        final_hidden = torch.cat([final_hidden, hidden[-1][w : w + 1, :kept, :]], dim=1)
        target_hidden = extract_context_feature([h[w : w + 1, :kept, :] for h in hidden], drafter.target_layer_ids)
        accept_lens.append(kept)
        if torch.isin(output_ids[0, start - kept : start], stop).any():
            break

    generated = output_ids[:, n_in:start]
    hit = torch.isin(generated[0], stop).nonzero()
    if hit.numel():
        generated = generated[:, : int(hit[0]) + 1]
    return generated, accept_lens
