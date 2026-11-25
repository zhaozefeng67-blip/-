# -*- coding: utf-8 -*-
"""Measurement layer for PBSG plan sections 7.2, 7.3, 7.4 (and 7.5 memory).

Boundary rules that must stay consistent (section 7.4: "各阶段边界必须定义一致，避免重复计时"):

  T_epoch = T_data + T_graph + T_fwd + T_bwd + T_opt

  T_data   time to pull the next batch from the loader (may overlap nothing; measured
           after a synchronise so it is not hidden behind GPU work)
  T_graph  builder() only
  T_fwd    backbone forward only
  T_bwd    loss construction + backward
  T_opt    optimizer step

Every stage is bounded by ``cuda.synchronize`` so the split is exact and stages do not
double count.  This makes training slower than an un-instrumented run, which is why the
decomposition is measured in a separate short pass (3 epochs) rather than in the main
accuracy sweep.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q))


# ------------------------------------------------------------------ 7.2 structure
def structural_stats(builder, A, N, cand_stats=None):
    """Every quantity listed in section 7.2, taken from the operator actually built."""
    if A.is_sparse:
        idx = A.coalesce().indices()
        rows = idx[0]
        cols = idx[1]
        vals = A.coalesce().values()
        nz = vals.abs() > 0
        rows, cols = rows[nz], cols[nz]
        edges_all = int(rows.numel())
        nonself = rows != cols
        edges = int(nonself.sum())
        out_deg = torch.bincount(rows, minlength=N)
        adj_bytes = int(A.coalesce().indices().numel() * A.coalesce().indices().element_size()
                        + A.coalesce().values().numel()
                        * A.coalesce().values().element_size())
        storage_format = 'COO'
    else:
        nz = A.abs() > 0
        edges_all = int(nz.sum())
        diag = torch.diagonal(nz).sum()
        edges = int(edges_all - diag)
        out_deg = nz.sum(dim=1)
        adj_bytes = int(A.numel() * A.element_size())
        storage_format = 'dense'

    st = builder.last_stats
    prior = int(st.get('prior_edges', 0))
    budget = int(st.get('budget_edges', 0))
    scored = int(st.get('scored_pairs', 0))
    denom = prior + budget
    out = {
        'edges_with_self': edges_all,
        'edges_without_self': edges,
        'avg_out_degree': float(edges) / float(N),
        'max_out_degree': int(out_deg.max()),
        'graph_density': float(edges) / float(N * (N - 1)),
        'scored_pairs': scored,
        'prior_candidate_count': prior,
        'budget_edge_count': budget,
        'budget_edge_ratio': (float(budget) / float(denom)) if denom else 0.0,
        'adjacency_bytes': adj_bytes,
        'adjacency_storage_format': storage_format,
    }
    if cand_stats:
        ps = cand_stats.get('prior_support', {})
        fp = cand_stats.get('external_pool', {})
        out['ci_edges_total'] = int(ps.get('ci_edges_total', 0))
        out['ci_size_mean'] = float(ps.get('ci_size_mean', 0.0))
        out['k_eff'] = int(ps.get('K_eff', 0))
        out['fi_edges_total'] = int(fp.get('fi_edges_total', 0))
        out['prior_coverage'] = float(cand_stats.get('prior_coverage', {}).get('coverage', 0.0))
        out['candidate_storage_bytes'] = int(
            cand_stats.get('storage', {}).get('cand_bytes', 0)
            + cand_stats.get('storage', {}).get('pool_bytes', 0))
    return out


# ------------------------------------------------------------------ 7.3 graph build
def time_builder(builder, dev, warmup=10, reps=100):
    """Graph-builder-only latency with a FIXED embedding and cached indices (7.3)."""
    emb = builder.emb
    ctx = torch.no_grad()
    with ctx:
        for _ in range(warmup):
            builder()
        _sync(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            builder()
            _sync(dev)
            ts.append((time.perf_counter() - t0) * 1000.0)
    graph_peak = None
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            builder()
        _sync(dev)
        graph_peak = float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return {
        'graph_ms_mean': float(np.mean(ts)),
        'graph_ms_median': _pct(ts, 50),
        'graph_ms_p95': _pct(ts, 95),
        'graph_build_reps': reps,
        'graph_builder_peak_mb': graph_peak,
        'embedding_fixed': emb is not None,
    }


# ------------------------------------------------------------------ 7.3/7.4 latency
def time_inference(model, builder, x_one, dev, warmup=10, reps=100):
    """Two variants required by 7.3: cached-graph vs graph-build-inclusive, batch = 1."""
    out = {}
    model.eval()
    with torch.no_grad():
        A = builder()
        for _ in range(warmup):
            model(x_one, A)
        _sync(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x_one, A)
            _sync(dev)
            ts.append((time.perf_counter() - t0) * 1000.0)
        out['infer_cached_ms_mean'] = float(np.mean(ts))
        out['infer_cached_ms_median'] = _pct(ts, 50)
        out['infer_cached_ms_p95'] = _pct(ts, 95)

        for _ in range(warmup):
            A2 = builder()
            model(x_one, A2)
        _sync(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            A2 = builder()
            model(x_one, A2)
            _sync(dev)
            ts.append((time.perf_counter() - t0) * 1000.0)
        out['infer_build_ms_mean'] = float(np.mean(ts))
        out['infer_build_ms_median'] = _pct(ts, 50)
        out['infer_build_ms_p95'] = _pct(ts, 95)
        out['infer_reps'] = reps

        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            A3 = builder()
            model(x_one, A3)
            _sync(dev)
            out['infer_peak_mb'] = float(torch.cuda.max_memory_allocated() / 2 ** 20)
        else:
            out['infer_peak_mb'] = None
    return out


# ------------------------------------------------------------------ 7.4 batch probe
def max_runnable_batch(model, builder, ds_one, dev, cands=(16, 32, 64, 128, 256)):
    """Largest batch that completes a full train step; records the OOM stage (7.6)."""
    best = 0
    stages = {}
    opt = torch.optim.Adam(list(model.parameters())
                           + ([builder.emb] if builder.emb is not None else []), lr=1e-3)
    lossf = nn.L1Loss(reduction='none')
    x0, y0, m0 = ds_one[0]
    for bs in cands:
        xb = x0.unsqueeze(0).repeat(bs, 1, 1).to(dev)
        yb = y0.unsqueeze(0).repeat(bs, 1, 1).to(dev)
        mb = m0.unsqueeze(0).repeat(bs, 1, 1).to(dev)
        stage = 'graph'
        try:
            model.train()
            opt.zero_grad(set_to_none=True)
            A = builder()
            stage = 'forward'
            pred = model(xb, A)
            stage = 'backward'
            l = lossf(pred, yb)
            l = (l * mb).sum() / mb.sum().clamp(min=1.0)
            l.backward()
            stage = 'optimizer'
            opt.step()
            _sync(dev)
            best = bs
            stages[bs] = 'ok'
        except torch.cuda.OutOfMemoryError:
            stages[bs] = 'oom@' + stage
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'memory' in msg:
                stages[bs] = 'oom@' + stage
            else:
                stages[bs] = 'err@' + stage
            torch.cuda.empty_cache()
            break
        finally:
            del xb, yb, mb
    return {'max_batch_size': best, 'batch_probe': stages,
            'batch_candidates': list(cands)}


# ------------------------------------------------------------------ 7.4 decomposition
def epoch_decomposition(model, builder, dl, dev, epochs=3):
    """T_epoch = T_data + T_graph + T_fwd + T_bwd + T_opt, all bounded by synchronise."""
    opt = torch.optim.Adam(list(model.parameters())
                           + ([builder.emb] if builder.emb is not None else []), lr=1e-3)
    lossf = nn.L1Loss(reduction='none')
    acc = {k: 0.0 for k in ('data', 'graph', 'fwd', 'bwd', 'opt')}
    n = 0
    model.train()
    for _ in range(epochs):
        it = iter(dl)
        while True:
            t0 = time.perf_counter()
            try:
                xb, yb, mb = next(it)
            except StopIteration:
                break
            _sync(dev)
            t1 = time.perf_counter()
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            A = builder()
            _sync(dev)
            t2 = time.perf_counter()
            pred = model(xb, A)
            _sync(dev)
            t3 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            l = lossf(pred, yb)
            l = (l * mb).sum() / mb.sum().clamp(min=1.0)
            l.backward()
            _sync(dev)
            t4 = time.perf_counter()
            opt.step()
            _sync(dev)
            t5 = time.perf_counter()
            acc['data'] += t1 - t0
            acc['graph'] += t2 - t1
            acc['fwd'] += t3 - t2
            acc['bwd'] += t4 - t3
            acc['opt'] += t5 - t4
            n += 1
            if n >= 40 * epochs:
                break
    tot = sum(acc.values())
    out = {'decomp_steps': n, 'decomp_total_sec': tot}
    for k, v in acc.items():
        out['decomp_%s_sec' % k] = v
        out['decomp_%s_pct' % k] = 100.0 * v / tot if tot else 0.0
    out['throughput_samples_per_sec'] = (n * dl.batch_size / tot) if tot else None
    return out


# ------------------------------------------------------------------ 7.5 memory
def peak_train_memory(dev):
    if dev.type != 'cuda':
        return None
    return float(torch.cuda.max_memory_allocated() / 2 ** 20)
