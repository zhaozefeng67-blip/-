# -*- coding: utf-8 -*-
"""Pareto / cost-accuracy figures (PBSG plan section 7.7).

Deliberately does NOT collapse accuracy, time and memory into one score: it plots the
frontier ``f = (MAE, graph build time, epoch time, memory)`` and leaves the trade-off
visible.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from pbsg import paths

LABEL = {
    'road': 'A Road-Fixed', 'dense': 'B Dense-Adaptive', 'topk': 'C Global-TopK',
    'fixed_cand': 'F Fixed-Candidate', 'rand_bud': 'G Random-Budget',
}


def mlabel(c):
    k = c['kind']
    if k == 'pbsg':
        return 'D PBSG-m0' if c['m'] == 0 else 'E PBSG-m%d' % c['m']
    return LABEL.get(k, k)


def load(runs):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, '*', 'result.json'))):
        with open(p) as f:
            r = json.load(f)
        c = r['config']
        pr = r.get('profile') or {}
        rows.append({
            'dataset': c['dataset'], 'kind': c['kind'], 'K': c['K'], 'M': c['M'],
            'm': c['m'], 'seed': c['seed'], 'label': mlabel(c),
            'mae': r['metrics']['test']['mae'],
            'rmse': r['metrics']['test']['rmse'],
            'graph_ms': (r['graph_sec_per_epoch'] / max(1e-9, r['sec_per_epoch'])) * 1000.0,
            'graph_sec_ep': r['graph_sec_per_epoch'],
            'sec_ep': r['sec_per_epoch'],
            'lat_ms': pr.get('latency_p50_ms'),
            'mem_mb': pr.get('infer_peak_mb') or r.get('peak_train_mb'),
            'edges': r['A_edges'],
        })
    # average seeds
    agg = defaultdict(list)
    for r in rows:
        agg[(r['dataset'], r['kind'], r['K'], r['M'], r['m'], r['label'])].append(r)
    out = []
    for k, rs in agg.items():
        d = dict(rs[0])
        for f in ('mae', 'rmse', 'graph_sec_ep', 'sec_ep', 'lat_ms', 'mem_mb', 'edges'):
            vals = [x[f] for x in rs if x[f] is not None and np.isfinite(x[f])]
            d[f] = float(np.mean(vals)) if vals else None
        d['n'] = len(rs)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(paths.RUNS))
    ap.add_argument('--outdir', default=str(paths.FIGS))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = load(a.runs)
    if not rows:
        raise SystemExit('no results in %s' % a.runs)
    dsets = sorted(set(r['dataset'] for r in rows))
    print('datasets: %s   configs: %d' % (dsets, len(rows)))
    made = []

    # 1) budget m vs MAE / RMSE
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ds in dsets:
        pts = sorted([r for r in rows if r['dataset'] == ds and r['kind'] == 'pbsg'],
                     key=lambda r: r['m'])
        p = [r for r in pts if r['K'] == 32 and r['M'] == 64]
        if not p:
            continue
        ms = [r['m'] for r in p]
        ax[0].plot(ms, [r['mae'] for r in p], marker='o', label=ds)
        ax[1].plot(ms, [r['rmse'] for r in p], marker='o', label=ds)
    for i, t in enumerate(('MAE', 'RMSE')):
        ax[i].set_xlabel('budget m'); ax[i].set_ylabel(t)
        ax[i].set_title('%s vs budget m (K=32, M=64)' % t)
        ax[i].grid(alpha=.3); ax[i].legend()
    fig.tight_layout(); f1 = os.path.join(a.outdir, 'pareto_m_vs_mae.png')
    fig.savefig(f1, dpi=140); plt.close(fig); made.append(f1)

    # 2) accuracy vs graph-construction cost, all methods
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in dsets:
        for r in rows:
            if r['dataset'] != ds or r['mae'] is None:
                continue
            ax.scatter(r['graph_sec_ep'] * 1000.0, r['mae'], s=46)
            ax.annotate(r['label'], (r['graph_sec_ep'] * 1000.0, r['mae']),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xlabel('graph build time per epoch (ms, log)')
    ax.set_ylabel('MAE')
    ax.set_xscale('log')
    ax.set_title('MAE vs graph construction cost')
    ax.grid(alpha=.3)
    fig.tight_layout(); f2 = os.path.join(a.outdir, 'pareto_mae_vs_graphcost.png')
    fig.savefig(f2, dpi=140); plt.close(fig); made.append(f2)

    # 3) N vs graph construction time (needs >= 2 node counts)
    n_of = {}
    for ds in dsets:
        mp = paths.PROC / ds / 'meta.json'
        if mp.exists():
            with open(mp) as f:
                n_of[ds] = json.load(f)['N']
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab in sorted(set(r['label'] for r in rows)):
        pts = [r for r in rows if r['label'] == lab and r['graph_sec_ep'] is not None]
        xs = sorted(set(r['dataset'] for r in pts), key=lambda d: n_of.get(d, 0))
        if len(xs) < 2:
            continue
        nmap = {}
        for p in pts:
            nmap.setdefault(p['dataset'], []).append(p['graph_sec_ep'])
        ax.plot(range(len(xs)), [np.mean(nmap[x]) * 1000 for x in xs], marker='o', label=lab)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(['%s\nN=%d' % (x, n_of.get(x, -1)) for x in xs])
    ax.set_ylabel('graph build time per epoch (ms)')
    ax.set_title('node count vs graph construction time')
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); f3 = os.path.join(a.outdir, 'pareto_n_vs_graphcost.png')
    fig.savefig(f3, dpi=140); plt.close(fig); made.append(f3)

    # 4) accuracy vs epoch time, and vs memory if available
    for fld, fname, xlab in (('sec_ep', 'pareto_mae_vs_epochtime', 'sec / epoch'),
                             ('mem_mb', 'pareto_mae_vs_memory', 'peak CUDA (MB)')):
        fig, ax = plt.subplots(figsize=(7, 5))
        got = False
        for r in rows:
            if r['mae'] is None or r[fld] is None:
                continue
            got = True
            ax.scatter(r[fld], r['mae'], s=46)
            ax.annotate(r['label'], (r[fld], r['mae']), textcoords='offset points',
                        xytext=(4, 4), fontsize=7)
        if not got:
            plt.close(fig)
            continue
        ax.set_xlabel(xlab); ax.set_ylabel('MAE')
        ax.set_title('MAE vs %s' % xlab)
        ax.grid(alpha=.3)
        fig.tight_layout(); fp = os.path.join(a.outdir, fname + '.png')
        fig.savefig(fp, dpi=140); plt.close(fig); made.append(fp)

    for p in made:
        print('wrote', p)


if __name__ == '__main__':
    main()
