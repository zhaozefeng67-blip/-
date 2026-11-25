# -*- coding: utf-8 -*-
"""Road-prior support C_i (section 3.3) and bounded prior-external pool F_i (section 3.5).

Both are built OFFLINE from a sparse CSR road adjacency and cached, exactly once per
(K, M, seed).  No N x N matrix is ever materialised, and F_i is drawn by bounded
rejection sampling with a per-node deterministic RNG so the pool is reproducible.
"""
import json
import os

import numpy as np


def csr_from_edges(src, dst, N):
    """Directed CSR (row = out-edges), rows sorted by target id."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    counts = np.bincount(src, minlength=N)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, dst


def build_prior_candidates(indptr, indices, K, max_hops=4):
    """C_i = bounded BFS support, hop-first ordering, deterministic tie-break by node id.

    Section 3.3: if a node's 1-hop road neighbours already exceed K they are NOT
    silently dropped -- the node keeps all of them and K_eff grows to cover it.
    """
    N = len(indptr) - 1
    lists = []
    hop_hist = {}
    n_eff_gt_k = 0
    for i in range(N):
        h1 = np.unique(indices[indptr[i]:indptr[i + 1]])
        h1 = h1[h1 != i]
        if h1.size > K:
            sel = [int(v) for v in h1]
            n_eff_gt_k += 1
            hop_hist[1] = hop_hist.get(1, 0) + len(sel)
            lists.append(sel)
            continue
        seen = set(int(v) for v in h1)
        seen.add(i)
        sel = [int(v) for v in h1]
        hop_hist[1] = hop_hist.get(1, 0) + len(sel)
        frontier = [int(v) for v in h1]
        for h in range(2, max_hops + 1):
            if len(sel) >= K or not frontier:
                break
            nxt = []
            for u in frontier:
                for v in indices[indptr[u]:indptr[u + 1]]:
                    v = int(v)
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            if not nxt:
                break
            nxt.sort()
            need = K - len(sel)
            take = nxt[:need]
            sel.extend(take)
            hop_hist[h] = hop_hist.get(h, 0) + len(take)
            frontier = nxt
        lists.append(sel)

    Kmax = max(len(s) for s in lists)
    C = np.full((N, Kmax), -1, dtype=np.int32)
    for i, s in enumerate(lists):
        C[i, :len(s)] = s
    K_eff = Kmax
    stats = {
        'K': int(K),
        'K_eff': int(K_eff),
        'max_hops': int(max_hops),
        'nodes_with_1hop_over_K': int(n_eff_gt_k),
        'ci_size_min': int(min(len(s) for s in lists)),
        'ci_size_mean': float(np.mean([len(s) for s in lists])),
        'ci_size_max': int(K_eff),
        'ci_edges_total': int(sum(len(s) for s in lists)),
        'hop_composition': {str(k): int(v) for k, v in sorted(hop_hist.items())},
        'ordering': 'hop ascending, then node id ascending (deterministic)',
    }
    return C, K_eff, stats


def build_external_pool(N, C, M, seed=2026, max_tries=64):
    """F_i: at most M prior-external candidates per node, fixed seed, no N-wide scan."""
    Kmax = C.shape[1]
    F = np.full((N, M), -1, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    saturated = 0
    for i in range(N):
        excl = set([i])
        row = C[i]
        excl.update(int(v) for v in row[row >= 0])
        avail = N - len(excl)
        m = int(min(M, avail))
        if m <= 0:
            continue
        if m == avail:
            saturated += 1
        rng = np.random.default_rng([int(seed), int(i)])
        chosen = set()
        got = []
        tries = 0
        while len(got) < m and tries < max_tries:
            batch = max(2 * (m - len(got)), 16)
            draw = rng.integers(0, N, size=batch)
            for v in draw.tolist():
                if v in excl or v in chosen:
                    continue
                chosen.add(v)
                got.append(int(v))
                if len(got) == m:
                    break
            tries += 1
        F[i, :len(got)] = got
        counts[i] = len(got)

    stats = {
        'M': int(M),
        'seed': int(seed),
        'fi_size_min': int(counts.min()),
        'fi_size_mean': float(counts.mean()),
        'fi_size_max': int(counts.max()),
        'nodes_saturated_by_availability': int(saturated),
        'fi_edges_total': int(counts.sum()),
        'method': 'bounded rejection sampling, per-node rng = [seed, node]',
        'nodes_not_filled': int((counts < int(min(M, N - 1))).sum()),
    }
    return F, counts, stats


def prior_coverage(C, src, dst):
    """Fraction of released road edges whose target lies in the source's C_i set."""
    N = C.shape[0]
    covered = 0
    for u, v in zip(src.tolist(), dst.tolist()):
        row = C[u]
        if (row == v).any():
            covered += 1
    return {'road_edges': int(len(src)),
            'covered': int(covered),
            'coverage': float(covered / max(1, len(src)))}


def save_candidates(path, C, F, stats):
    np.savez(path, C=C, F=F)
    st = dict(stats)
    st['storage'] = {'cand_bytes': int(C.nbytes), 'pool_bytes': int(F.nbytes),
                     'cache_file_bytes': int(os.path.getsize(str(path)))}
    with open(str(path).replace('.npz', '_stats.json'), 'w') as f:
        json.dump(st, f, indent=2)
