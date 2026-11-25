# -*- coding: utf-8 -*-
"""D1: load raw PeMS parquet, audit, split on the time axis, fit train-only scaler, persist.

Protocol notes (PBSG plan sections 5.4-5.6):

* the split is made on the ORIGINAL time axis first (60/20/20), windows are then
  generated inside each split only, so no input or target crosses a boundary;
* the scaler is fitted on train-split valid observations only;
* zeros are NOT assumed to be missing.  They are audited and reported, and the
  mask defaults to "observed everywhere" (``zero_as_missing=False``);
* the road graph comes from the released sparse edge list (``from_id/to_id/cost``)
  and is used as a directed out-edge graph, following the row-wise convention in
  section 3.2.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

from . import paths

# The STSGCN parquet release keeps the raw station ids for PEMS03/PEMS07 while the series
# parquet has no id column, so the id -> array-index mapping is lost.  For those we fall
# back to an already-aligned Graph-WaveNet-format bundle: (T, N, C) flow plus a dense
# distance matrix whose row/column order IS the series node axis.
GWN_BUNDLES = {
    'PEMS07': ('pems07_gwn.npz', 'adj_PEMS07_distance.pkl'),
}


def _load_raw_gwn(dataset):
    npz_name, dist_name = GWN_BUNDLES[dataset]
    z = np.load(paths.RAW / npz_name)
    a = z['data']
    if a.ndim == 3:
        a = a[:, :, 0]
    X = np.asarray(a, dtype=np.float32)
    with open(paths.RAW / dist_name, 'rb') as f:
        D = pickle.load(f, encoding='latin1')
    iu, ju = np.nonzero(D)
    e = pd.DataFrame({'from_id': iu.astype(np.int64), 'to_id': ju.astype(np.int64),
                      'cost': D[iu, ju].astype(np.float64)})
    meta = {'N': int(X.shape[1]), 'T': int(X.shape[0]), 'channels': ['flow'],
            'edges': int(len(e)),
            'source': 'Graph-WaveNet-format bundle: %s + %s' % (npz_name, dist_name)}
    return X, e, meta


def _remap_sensor_ids(e, dataset, raw_meta):
    """Map real station ids onto the series axis when a sensors table is released.

    PEMS03's edge list uses Caltrans station ids while the series parquet has no id
    column; ``PEMS03_sensors.parquet`` supplies the ``node_id <-> sensor_id`` table and
    covers exactly the id set used by the edges.  Section 5.6: never guess the mapping.
    """
    p = paths.RAW / ('%s_sensors.parquet' % dataset)
    if not p.exists():
        return e, raw_meta
    s = pd.read_parquet(p)
    if not {'node_id', 'sensor_id'}.issubset(set(s.columns)):
        return e, raw_meta
    m = dict(zip(s['sensor_id'].tolist(), s['node_id'].tolist()))
    e = e.copy()
    e['from_id'] = e['from_id'].map(m)
    e['to_id'] = e['to_id'].map(m)
    if e['from_id'].isna().any() or e['to_id'].isna().any():
        raise ValueError('%s: sensors table does not cover the edge id set' % dataset)
    e['from_id'] = e['from_id'].astype('int64')
    e['to_id'] = e['to_id'].astype('int64')
    info = dict(raw_meta or {})
    info['sensor_id_remap'] = {'sensors': int(len(m)), 'source': p.name}
    return e, info


def load_raw(dataset):
    """Return (X float32 (T,N), edges DataFrame, raw_meta dict or None)."""
    if dataset in GWN_BUNDLES and (paths.RAW / GWN_BUNDLES[dataset][0]).exists():
        return _load_raw_gwn(dataset)

    sp = paths.RAW / ('%s_series.parquet' % dataset)
    ep = paths.RAW / ('%s_edges.parquet' % dataset)
    if not sp.exists():
        raise FileNotFoundError('missing series: %s' % sp)
    if not ep.exists():
        raise FileNotFoundError('missing edges: %s' % ep)

    s = pd.read_parquet(sp)
    if 'flow' not in s.columns:
        raise KeyError('%s has no flow column: %s' % (dataset, list(s.columns)))
    X = np.stack(s['flow'].to_numpy()).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError('%s flow contains non-finite values' % dataset)

    tcol = 'timestep' if 'timestep' in s.columns else None
    if tcol is not None:
        t = s[tcol].to_numpy()
        if not np.array_equal(t, np.arange(len(t))):
            raise ValueError('%s timestep axis is not 0..T-1' % dataset)

    e = pd.read_parquet(ep)
    for c in ('from_id', 'to_id'):
        if c not in e.columns:
            raise KeyError('%s edges missing %s' % (dataset, c))

    meta_path = paths.RAW / 'metadata.json'
    raw_meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            raw_meta = json.load(f).get('datasets', {}).get(dataset)
    e, raw_meta = _remap_sensor_ids(e, dataset, raw_meta)
    return X, e, raw_meta


def check_node_ids(dataset, X, e):
    """The edge node ids must index the series axis 0..N-1 exactly (section 5.6)."""
    N = X.shape[1]
    ids = np.unique(np.concatenate([e['from_id'].to_numpy(), e['to_id'].to_numpy()]))
    if int(ids.min()) < 0 or int(ids.max()) >= N:
        raise ValueError(
            '%s edge ids span [%d, %d] but the series axis has N=%d. The released '
            'sensor-id -> array-index mapping is required and %s_sensors.parquet is '
            'unavailable. Refusing to guess the mapping (section 5.6).'
            % (dataset, ids.min(), ids.max(), N, dataset))
    return {
        'edge_id_min': int(ids.min()),
        'edge_id_max': int(ids.max()),
        'unique_edge_nodes': int(len(ids)),
        'nodes_without_edges': int(N - len(ids)),
    }


def audit_series(X, dataset):
    """Zero / outage audit.  Zeros are reported, never silently dropped."""
    z = (X == 0.0)
    node_zero = z.mean(axis=0)
    return {
        'dataset': dataset,
        'T': int(X.shape[0]),
        'N': int(X.shape[1]),
        'zero_frac_all': float(z.mean()),
        'node_zero_frac_min': float(node_zero.min()),
        'node_zero_frac_median': float(np.median(node_zero)),
        'node_zero_frac_max': float(node_zero.max()),
        'nodes_zero_frac_gt_50pct': int((node_zero > 0.5).sum()),
        'value_min': float(X.min()),
        'value_max': float(X.max()),
        'node_zero_frac_argmax': int(node_zero.argmax()),
    }


def audit_degree(src, dst, N, tag):
    out_deg = np.bincount(src, minlength=N)
    in_deg = np.bincount(dst, minlength=N)
    return {
        'tag': tag,
        'n_edges': int(len(src)),
        'self_loops': int((src == dst).sum()),
        'avg_out_degree': float(out_deg.mean()),
        'max_out_degree': int(out_deg.max()),
        'out_degree_zero': int((out_deg == 0).sum()),
        'in_degree_zero': int((in_deg == 0).sum()),
        'either_degree_zero': int(((out_deg + in_deg) == 0).sum()),
    }


def normalize_edges(src, dst, cost, N, symmetrize=True):
    """Dedupe, drop self loops, optionally symmetrize; record every step (sections 3.2/5.6)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    cost = np.asarray(cost, dtype=np.float64)
    n_raw = len(src)
    before = audit_degree(src, dst, N, 'as_released_directed')

    keep = src != dst
    src, dst, cost = src[keep], dst[keep], cost[keep]
    n_after_selfloop = len(src)

    stack = np.stack([src, dst], axis=1)
    _, first = np.unique(stack, axis=0, return_index=True)
    first = np.sort(first)
    src, dst, cost = src[first], dst[first], cost[first]
    n_dedup = len(src)

    if symmetrize:
        rev_src, rev_dst, rev_cost = dst.copy(), src.copy(), cost.copy()
        s2 = np.concatenate([src, rev_src])
        d2 = np.concatenate([dst, rev_dst])
        c2 = np.concatenate([cost, rev_cost])
        stack2 = np.stack([s2, d2], axis=1)
        _, first2 = np.unique(stack2, axis=0, return_index=True)
        first2 = np.sort(first2)
        src, dst, cost = s2[first2], d2[first2], c2[first2]

    after = audit_degree(src, dst, N, 'used_for_C_i')
    info = {
        'symmetrized': bool(symmetrize),
        'rows_as_released': int(n_raw),
        'rows_after_selfloop_removal': int(n_after_selfloop),
        'rows_after_dedupe': int(n_dedup),
        'rows_final': int(len(src)),
        'duplicate_rows_removed': int(n_after_selfloop - n_dedup),
        'before': before,
        'after': after,
    }
    return src, dst, cost, info


def split_points(T, ratios=(0.6, 0.2, 0.2)):
    """Boundaries on the ORIGINAL time axis (section 5.4)."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError('ratios must sum to 1, got %s' % (ratios,))
    n_tr = int(round(T * ratios[0]))
    n_va = int(round(T * ratios[1]))
    return {'train': (0, n_tr), 'val': (n_tr, n_tr + n_va), 'test': (n_tr + n_va, T)}


def window_index(a, b, tin, tout):
    """Window start indices fully contained in [a, b).  No cross-boundary windows."""
    last = b - tin - tout
    if last < a:
        return np.empty(0, dtype=np.int64)
    return np.arange(a, last + 1, dtype=np.int64)


def fit_scaler(Xtr, mask=None):
    """Per-node mean/std from train split only."""
    if mask is None:
        mask = np.ones_like(Xtr, dtype=bool)
    cnt = mask.sum(axis=0).astype(np.float64)
    if (cnt == 0).any():
        raise ValueError('train split has nodes with no valid observation')
    mean = (Xtr * mask).sum(axis=0) / cnt
    var = ((Xtr - mean) ** 2 * mask).sum(axis=0) / cnt
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build(dataset, tin=12, tout=12, ratios=(0.6, 0.2, 0.2), zero_as_missing=False,
          symmetrize=True):
    X, e, raw_meta = load_raw(dataset)
    node_audit = check_node_ids(dataset, X, e)
    s_audit = audit_series(X, dataset)

    T, N = X.shape
    src = e['from_id'].to_numpy()
    dst = e['to_id'].to_numpy()
    cost = e['cost'].to_numpy() if 'cost' in e.columns else np.zeros(len(src))
    src, dst, cost, e_audit = normalize_edges(src, dst, cost, N, symmetrize)
    e_audit['cost_unit'] = ('meters_like' if np.median(cost) > 20 else 'km_like')
    e_audit['cost_median'] = float(np.median(cost))
    e_audit['cost_min'] = float(cost.min())
    e_audit['cost_max'] = float(cost.max())
    e_audit['dataset'] = dataset

    sp = split_points(T, ratios)
    windows = {k: window_index(v[0], v[1], tin, tout) for k, v in sp.items()}
    if len(windows['train']) == 0:
        raise ValueError('%s: train split too short for tin=%d tout=%d' % (dataset, tin, tout))

    # mask: observed everywhere unless zeros are explicitly declared missing
    mask = np.ones((T, N), dtype=bool) if not zero_as_missing else (X != 0.0)
    tr_a, tr_b = sp['train']
    mean, std = fit_scaler(X[tr_a:tr_b], mask[tr_a:tr_b])

    d = paths.processed_dir(dataset)
    np.save(d / 'x.npy', X)
    np.save(d / 'mask.npy', mask.astype(np.uint8))
    np.savez(d / 'scaler.npz', mean=mean, std=std)
    np.savez(d / 'road.npz',
             src=src.astype(np.int32),
             dst=dst.astype(np.int32),
             cost=cost.astype(np.float64))

    meta = {
        'dataset': dataset,
        'tin': tin,
        'tout': tout,
        'ratios': list(ratios),
        'T': T,
        'N': N,
        'split_bounds': {k: [int(v[0]), int(v[1])] for k, v in sp.items()},
        'windows': {k: int(len(v)) for k, v in windows.items()},
        'window_arrays': {k: v.tolist() for k, v in windows.items()},
        'zero_as_missing': bool(zero_as_missing),
        'scaler': {'fit_on': 'train split only',
                   'mean_mean': float(mean.mean()),
                   'std_mean': float(std.mean())},
        'series_audit': s_audit,
        'edge_audit': e_audit,
        'node_id_audit': node_audit,
        'raw_meta': raw_meta,
    }
    with open(d / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta
