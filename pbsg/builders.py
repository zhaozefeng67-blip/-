# -*- coding: utf-8 -*-
"""Graph builders A-G (PBSG plan section 6.2 / 6.3).

Every method returns an ``(N, N)`` operator consumed by ONE fixed backbone, and every
method receives the *same* score definition ``s_ij = ReLU(e_i^T e_j)`` so that the only
difference between methods is the SUPPORT on which scores are computed and normalised.

Weight convention (section 3.2 / 6.1): row i holds node i's outgoing supports, and the
propagation uses ``A @ H`` so that node i aggregates from the columns j it points to.
Self loops follow one unified rule: score 0, included in the row softmax.

Fairness / cost conventions:

* the candidate indices C_i and F_i are built OFFLINE and cached, so the online cost is
  scoring + top-m + normalisation only (section 7.3);
* methods whose support does not depend on the scores (``fixed_cand``, ``rand_bud``, and
  ``pbsg`` with m = 0) cache the SPARSITY PATTERN as well -- only the values are recomputed
  each forward.  ``pbsg`` with m > 0 cannot: hard top-m makes the selected columns
  score-dependent, so its online cost legitimately includes index rebuilding.
"""
import numpy as np
import torch
import torch.nn as nn

KINDS = ('road', 'dense', 'topk', 'pbsg', 'fixed_cand', 'rand_bud')
LEARNED = ('dense', 'topk', 'pbsg', 'rand_bud')


def spmm(A, H):
    """``A @ H`` over the node axis; H is (B, C, N) or (B, C, N, T)."""
    if H.dim() == 3:
        B, C, N = H.shape
        x = H.reshape(B * C, N).t()
        y = torch.sparse.mm(A, x) if A.is_sparse else torch.mm(A, x)
        return y.t().reshape(B, C, N)
    if H.dim() != 4:
        raise ValueError('spmm expects 3D or 4D input, got %s' % (tuple(H.shape),))
    B, C, N, T = H.shape
    x = H.permute(0, 1, 3, 2).reshape(-1, N)
    y = torch.sparse.mm(A, x.t()) if A.is_sparse else torch.mm(A, x.t())
    y = y.t().reshape(B, C, T, N).permute(0, 1, 3, 2)
    return y.contiguous()


def row_softmax(scores, valid, temp=1.0):
    """Row softmax over a padded candidate matrix; invalid entries become weight 0."""
    s = torch.where(valid, scores, torch.full_like(scores, float('-inf')))
    w = torch.softmax(s / temp, dim=1)
    return torch.where(valid, w, torch.zeros_like(w))


class GraphBuilder(nn.Module):
    """Unified builder.  ``kind`` selects one of A-G."""

    def __init__(self, kind, N, K, M, m, d, road_indptr, road_indices,
                 cand_C=None, cand_F=None, seed=2026, k_topk=None, temp=1.0):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        self.N = int(N)
        self.K = int(K)
        self.M = int(M)
        self.m = int(m)
        self.temp = float(temp)
        self.d = int(d)
        self.k_topk = int(k_topk) if k_topk is not None else int(K) + int(m)
        self.last_stats = {}
        self._pat = None          # cached coalesced sparsity pattern (indices, inverse)
        self._row_cache = None    # cached row index tensor
        self._static = (kind in ('fixed_cand', 'rand_bud')
                        or (kind == 'pbsg' and self.m == 0))

        # prior road support kept as CSR buffers; the all-ones index is built once
        self.register_buffer('road_indptr', torch.as_tensor(road_indptr, dtype=torch.long))
        self.register_buffer('road_indices', torch.as_tensor(road_indices, dtype=torch.long))
        rows, cols = self._road_index()
        self.register_buffer('road_rows', torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer('road_cols', torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer('road_vals', self._road_weights())

        if kind in ('pbsg', 'fixed_cand', 'rand_bud'):
            assert cand_C is not None and cand_F is not None
            Cm = torch.as_tensor(cand_C, dtype=torch.long)
            Fm = torch.as_tensor(cand_F, dtype=torch.long)
            self.register_buffer('C_idx', Cm)
            self.register_buffer('C_valid', Cm >= 0)
            self.register_buffer('F_idx', Fm)
            self.register_buffer('F_valid', Fm >= 0)
            if kind == 'rand_bud':
                g = torch.Generator().manual_seed(int(seed))
                perm = torch.argsort(torch.rand(Cm.shape[0], Fm.shape[1], generator=g), dim=1)
                self.register_buffer('rand_sel', perm)

        self.emb = nn.Parameter(torch.empty(self.N, self.d)) if kind in LEARNED else None
        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb)

    # ---------------- prior road operator ----------------
    def _road_index(self):
        """Flat (row, col) index of the road operator, sorted so the COO is coalesced."""
        N = self.N
        rows = np.repeat(np.arange(N), np.diff(self.road_indptr.numpy()))
        cols = self.road_indices.numpy().copy()
        rows = np.concatenate([rows, np.arange(N)]).astype(np.int64)
        cols = np.concatenate([cols, np.arange(N)]).astype(np.int64)
        order = np.lexsort((cols, rows))
        return rows[order], cols[order]

    def _road_weights(self):
        N = self.N
        deg = np.zeros(N, dtype=np.float64)
        np.add.at(deg, self.road_rows.numpy(), 1.0)
        w = np.ones(len(self.road_rows), dtype=np.float64) / deg[self.road_rows.numpy()]
        return torch.as_tensor(w.astype(np.float32))

    def _road_operator(self):
        idx = torch.stack([self.road_rows, self.road_cols])
        return torch.sparse_coo_tensor(idx, self.road_vals, (self.N, self.N),
                                       is_coalesced=True)

    # ---------------- score helpers ----------------
    def _dense_scores(self):
        E = self.emb
        s = torch.relu(E @ E.t())
        s = s.masked_fill(torch.eye(self.N, device=s.device, dtype=torch.bool), 0.0)
        return s

    def _support(self):
        """Return (col, weight, valid) for the padded candidate layout."""
        E = self.emb
        N = self.N
        dev = E.device
        Ek = E[self.C_idx.clamp(min=0)]
        sC = torch.relu((E.unsqueeze(1) * Ek).sum(-1))
        sC = torch.where(self.C_valid, sC, torch.zeros_like(sC))
        cols = [self.C_idx]
        vals = [sC]
        valid = [self.C_valid]

        if self.kind in ('pbsg', 'rand_bud') and self.m > 0:
            Fk = E[self.F_idx.clamp(min=0)]
            sF = torch.relu((E.unsqueeze(1) * Fk).sum(-1))
            sF = torch.where(self.F_valid, sF, torch.full_like(sF, float('-inf')))
            if self.kind == 'rand_bud':
                sel = self.rand_sel[:, :self.m]
                sel_s = torch.gather(sF, 1, sel)
                sel_idx = torch.gather(self.F_idx, 1, sel)
                sel_valid = torch.isfinite(sel_s) & (sel_idx >= 0)
                sel_s = torch.where(sel_valid, sel_s, torch.zeros_like(sel_s))
            else:
                top = torch.topk(sF, k=min(self.m, sF.shape[1]), dim=1)
                sel_s, sel = top.values, top.indices
                sel_idx = torch.gather(self.F_idx, 1, sel)
                sel_valid = torch.isfinite(sel_s) & (sel_idx >= 0)
                sel_s = torch.where(sel_valid, sel_s, torch.zeros_like(sel_s))
            cols.append(torch.where(sel_valid, sel_idx, torch.zeros_like(sel_idx)))
            vals.append(sel_s)
            valid.append(sel_valid)

        cols.append(torch.arange(N, device=dev).unsqueeze(1))
        vals.append(torch.zeros(N, 1, device=dev))
        valid.append(torch.ones(N, 1, dtype=torch.bool, device=dev))

        col = torch.cat(cols, dim=1)
        sc = torch.cat(vals, dim=1)
        va = torch.cat(valid, dim=1)
        return col, row_softmax(sc, va, self.temp), va

    def _fixed_support(self):
        """F: same support as PBSG-m0 but the weights are fixed (no learned embedding)."""
        N = self.N
        dev = self.C_idx.device
        self_col = torch.arange(N, device=dev).unsqueeze(1)
        col = torch.cat([self.C_idx, self_col], dim=1)
        w = torch.cat([self.C_valid.to(torch.float32), torch.ones(N, 1, device=dev)], dim=1)
        va = torch.cat([self.C_valid,
                        torch.ones(N, 1, dtype=torch.bool, device=dev)], dim=1)
        return col, w / w.sum(dim=1, keepdim=True), va

    # ---------------- sparse assembly ----------------
    def _row_index(self, col):
        key = (tuple(col.shape), col.device)
        if self._row_cache is None or self._row_cache[0] != key:
            r = torch.arange(self.N, device=col.device).unsqueeze(1).expand_as(col)
            r = r.reshape(-1).contiguous()
            self._row_cache = (key, r)
        return self._row_cache[1]

    def _assemble(self, col, w):
        """Build A.  Cached pattern (coalesced, sorted) when the support is score-static."""
        row = self._row_index(col)
        col_safe = torch.where(col >= 0, col, row.reshape(col.shape)).reshape(-1)
        if self._static:
            if self._pat is None:
                idx = torch.stack([row, col_safe])
                uniq, inv = torch.unique(idx, dim=1, return_inverse=True)
                self._pat = (uniq.contiguous(), inv.reshape(-1))
            uniq, inv = self._pat
            vals = torch.zeros(uniq.shape[1], dtype=w.dtype, device=w.device)
            vals = vals.index_add(0, inv, w.reshape(-1))
            return torch.sparse_coo_tensor(uniq, vals, (self.N, self.N),
                                           is_coalesced=True)
        idx = torch.stack([row, col_safe])
        return torch.sparse_coo_tensor(idx, w.reshape(-1), (self.N, self.N)).coalesce()

    # ---------------- forward ----------------
    def forward(self):
        if self.kind == 'road':
            A = self._road_operator()
            self.last_stats = {'edges': int(A._nnz()),
                               'prior_edges': int(A._nnz()) - self.N,
                               'budget_edges': 0,
                               'scored_pairs': int(A._nnz())}
            return A
        if self.kind == 'dense':
            s = self._dense_scores()
            A = torch.softmax(s / self.temp, dim=1)
            self.last_stats = {'edges': int(self.N * self.N),
                               'prior_edges': 0, 'budget_edges': 0,
                               'scored_pairs': int(self.N * (self.N - 1))}
            return A
        if self.kind == 'topk':
            s = self._dense_scores()
            k = min(self.k_topk, self.N - 1)
            thresh = torch.topk(s, k=k, dim=1).values[:, -1:]
            keep = s >= thresh
            keep |= torch.eye(self.N, device=s.device, dtype=torch.bool)
            s = torch.where(keep, s, torch.full_like(s, float('-inf')))
            self.last_stats = {'edges': int(keep.sum()),
                               'prior_edges': 0, 'budget_edges': 0,
                               'scored_pairs': int(self.N * (self.N - 1))}
            return torch.softmax(s / self.temp, dim=1)
        col, w, va = (self._fixed_support() if self.kind == 'fixed_cand'
                      else self._support())
        A = self._assemble(col, w)
        kc = self.C_idx.shape[1]
        self.last_stats = {
            'edges': int(A._nnz()),
            'prior_edges': int(va[:, :kc].sum()),
            'budget_edges': int(va[:, kc:-1].sum()) if va.shape[1] > kc + 1 else 0,
            'scored_pairs': int(va.shape[0] * va.shape[1]),
        }
        return A
