"""Best-first choice of draft-refinement replacements under a verify-row budget.

Adapted from BASTION (arXiv 2605.29727; github.com/kaist-ai-osi-lab/BASTION, MIT, (c) 2026
Z Lab), with two changes forced by our verifier. The Qwen3.5 target has no tree-attention
mask, so every root-to-leaf path costs one whole verify row: the budget counts rows, and an
MTP-guided draft occupies the row its first token opens. A replacement (draft k, slot i,
token s) is a leaf: it copies draft k's tokens before slot i, places s at slot i, and the
drafter regenerates everything after it. Its score is
q_root(k) * prod_{i'<i} p_{i'}(draft | k) * p_i(s | k).
"""

import heapq

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def best_first_rows(root_probs, topk_vals, topk_idx, budget_rows):
    """Return the sibling rows (k, extension slot i, token) chosen within `budget_rows`.

    root_probs: [K] prior of each branch's t2 candidate. topk_vals/topk_idx: [K, L, c]
    descending per-slot probabilities and token ids of each branch's drafted continuation,
    where rank 0 is the drafted token and extension slot i is block slot i + 2.
    """
    root = _np(root_probs).reshape(-1)
    sv = _np(topk_vals)
    si = _np(topk_idx).astype(np.int64)
    n_branches, n_slots, kc = sv.shape
    spine_max = sv[:, :, 0]

    # A node is (k, slot i, rank r): r == -1 is the branch head, r == 0 its drafted token, r >= 1 a
    # sibling leaf. Scores only shrink along a path, so pops come in non-increasing order.
    nk, ni, nr, nrho = [], [], [], []
    heap = []

    def push(k, i, r, rho):
        nk.append(k)
        ni.append(i)
        nr.append(r)
        nrho.append(rho)
        heapq.heappush(heap, (-rho, len(nk) - 1, len(nk) - 1))

    for k in range(n_branches):
        push(k, -1, -1, float(root[k]))

    siblings = []
    rows_used = 0
    while heap:
        _, _, nid = heapq.heappop(heap)
        rho, k, i, r = nrho[nid], nk[nid], ni[nid], nr[nid]
        if r == -1:
            if rows_used >= budget_rows:
                continue
            rows_used += 1
            push(k, 0, 0, rho * spine_max[k, 0])
        elif r == 0:
            if i + 1 < n_slots:
                push(k, i + 1, 0, rho * spine_max[k, i + 1])
            if kc > 1 and spine_max[k, i] > 0.0:
                push(k, i, 1, rho / spine_max[k, i] * sv[k, i, 1])
        else:
            if rows_used >= budget_rows:
                continue
            rows_used += 1
            siblings.append((k, i, int(si[k, i, r])))
            if r + 1 < kc and sv[k, i, r] > 0.0:
                push(k, i, r + 1, rho / sv[k, i, r] * sv[k, i, r + 1])
    return siblings
