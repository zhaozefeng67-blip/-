# -*- coding: utf-8 -*-
"""Aggregate finished runs into the Stage-2 tables (section 8.5) and CSVs."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from pbsg import paths

METHOD_LABEL = {
    'road': 'A Road-Fixed',
    'dense': 'B Dense-Adaptive',
    'topk': 'C Global-TopK',
    'fixed_cand': 'F Fixed-Candidate',
    'rand_bud': 'G Random-Budget',
}


def label(r):
    k = r['config']['kind']
    if k == 'pbsg':
        return 'D PBSG-m0' if r['config']['m'] == 0 else 'E PBSG-m%d' % r['config']['m']
    return METHOD_LABEL.get(k, k)


def key(r):
    c = r['config']
    return (c['dataset'], label(r), '#'.join('%s=%s' % (a, c[a]) for a in ('K', 'M', 'm')))


def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float('nan'), float('nan'))
    return (v.mean(), v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(paths.RUNS))
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--min-seeds', type=int, default=1)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()

    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.runs, '*', 'result.json'))):
        with open(p) as f:
            r = json.load(f)
        if a.dataset and r['config']['dataset'] != a.dataset:
            continue
        groups[key(r)].append(r)

    acc_rows, eff_rows = [], []
    for k in sorted(groups):
        rs = groups[k]
        if len(rs) < a.min_seeds:
            continue
        ds, lab, sig = k
        mt = [r['metrics']['test'] for r in rs]
        mae, mae_sd = ms([m['mae'] for m in mt])
        rmse, rmse_sd = ms([m['rmse'] for m in mt])
        mape, _ = ms([m['mape'] for m in mt])
        h3, _ = ms([m.get('mae_h3') for m in mt])
        h6, _ = ms([m.get('mae_h6') for m in mt])
        h12, _ = ms([m.get('mae_h12') for m in mt])
        acc_rows.append({'dataset': ds, 'method': lab, 'sig': sig, 'n': len(rs),
                         'mae': mae, 'mae_sd': mae_sd, 'rmse': rmse, 'rmse_sd': rmse_sd,
                         'mape': mape, 'h3': h3, 'h6': h6, 'h12': h12})
        eff_rows.append({
            'dataset': ds, 'method': lab, 'sig': sig, 'n': len(rs),
            'A_edges': ms([r['A_edges'] for r in rs])[0],
            'sec_per_epoch': ms([r['sec_per_epoch'] for r in rs])[0],
            'graph_sec_per_epoch': ms([r['graph_sec_per_epoch'] for r in rs])[0],
            'wall_sec': ms([r['total_wall_sec'] for r in rs])[0],
            'params': ms([r['n_params'] for r in rs])[0],
            'epochs': ms([r['epochs_run'] for r in rs])[0],
        })

    acc_rows.sort(key=lambda r: (r['dataset'], r['mae']))
    print('=' * 108)
    print('ACCURACY (test, denormalised flow units, mean over seeds)')
    print('=' * 108)
    print('%-8s %-18s %-22s %2s %16s %7s %8s %8s %8s'
          % ('dataset', 'method', 'config', 'n', 'MAE(mean+-sd)', 'MAPE%', 'H3', 'H6', 'H12'))
    print('-' * 108)
    for r in acc_rows:
        print('%-8s %-18s %-22s %2d %8.3f+-%5.3f %7.2f %8.3f %8.3f %8.3f'
              % (r['dataset'], r['method'], r['sig'], r['n'], r['mae'], r['mae_sd'],
                 r['mape'], r['h3'], r['h6'], r['h12']))

    print()
    print('=' * 108)
    print('EFFICIENCY / GRAPH STRUCTURE')
    print('=' * 108)
    print('%-8s %-18s %-26s %7s %11s %11s %9s %8s'
          % ('dataset', 'method', 'config', 'A_edges', 's/epoch', 'graph_s/ep', 'wall_s', 'params'))
    print('-' * 108)
    for r in eff_rows:
        print('%-8s %-18s %-26s %7d %11.3f %11.4f %9.1f %8d'
              % (r['dataset'], r['method'], r['sig'], r['A_edges'], r['sec_per_epoch'],
                 r['graph_sec_per_epoch'], r['wall_sec'], r['params']))

    if a.csv:
        import csv as _csv
        with open(a.csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()))
            w.writeheader()
            w.writerows(acc_rows)
        with open(a.csv.replace('.csv', '_eff.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
            w.writeheader()
            w.writerows(eff_rows)
        print('\nwrote %s' % a.csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
