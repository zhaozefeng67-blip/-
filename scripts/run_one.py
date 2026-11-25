# -*- coding: utf-8 -*-
"""Run ONE configuration on ONE dataset (one seed).  Writes a single JSON result."""
import argparse
import json
import os
import sys
import time

from pbsg import engine, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--kind', required=True,
                    choices=['road', 'dense', 'topk', 'pbsg', 'fixed_cand', 'rand_bud'])
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--m', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cand-seed', type=int, default=2026)
    ap.add_argument('--d-emb', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--dilations', type=int, nargs='+', default=[1, 2, 4, 4])
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--k-topk', type=int, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()

    d = paths.processed_dir(a.dataset)
    cand = d / ('cand_K%d_M%d.npz' % (a.K, a.M))
    if a.kind in ('pbsg', 'fixed_cand', 'rand_bud') and not cand.exists():
        raise SystemExit('missing candidate cache: %s' % cand)

    # Section 6.2 asks C and E to match average non-self out-degree.  PBSG-m>0 has
    # mean|C_i| + m supports, not K + m, because most nodes saturate below K.  So
    # Global-TopK must use k = round(mean|C_i|) + m, not K + m.
    k_topk = a.k_topk
    if k_topk is None and a.kind == 'topk':
        st = d / ('cand_K%d_M%d_stats.json' % (a.K, a.M))
        if st.exists():
            with open(st) as f:
                ci_mean = json.load(f)['prior_support']['ci_size_mean']
            k_topk = int(round(ci_mean)) + a.m
        else:
            k_topk = a.K + a.m
        print('Global-TopK k aligned to mean|C_i| + m = %d (K+m would be %d)'
              % (k_topk, a.K + a.m), flush=True)

    cfg = {
        'dataset': a.dataset, 'kind': a.kind, 'K': a.K, 'M': a.M, 'm': a.m,
        'seed': a.seed, 'cand_seed': a.cand_seed, 'd_emb': a.d_emb,
        'temperature': a.temperature, 'channels': a.channels, 'dilations': a.dilations,
        'dropout': a.dropout, 'batch_size': a.batch_size,
        'lr': a.lr, 'weight_decay': a.weight_decay, 'max_epochs': a.max_epochs,
        'patience': a.patience, 'workers': a.workers, 'processed_dir': d,
        'k_topk': k_topk,
    }
    name = '%s_%s_K%d_M%d_m%d_s%d%s' % (a.dataset, a.kind, a.K, a.M, a.m, a.seed,
                                        ('_' + a.tag) if a.tag else '')
    outdir = a.outdir or (paths.RUNS / name)
    os.makedirs(outdir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(outdir / 'train.log', 'a') as f:
            f.write(msg + '\n')

    log('=' * 70)
    log('run %s  device=%s' % (name, 'cuda' if _cuda() else 'cpu'))
    log('=' * 70)
    t0 = time.perf_counter()
    out = engine.run(cfg, None, cand if cand.exists() else None, log=log)
    out['name'] = name
    out['total_wall_sec'] = time.perf_counter() - t0
    with open(outdir / 'result.json', 'w') as f:
        json.dump(out, f, indent=2)

    mt, mv = out['metrics']['test'], out['metrics']['val']
    log('-' * 70)
    log('TEST  MAE %.4f  RMSE %.4f  MAPE %.3f%%  H3 %.4f  H6 %.4f  H12 %.4f'
        % (mt['mae'], mt['rmse'], mt['mape'],
           mt.get('mae_h3', float('nan')), mt.get('mae_h6', float('nan')),
           mt.get('mae_h12', float('nan'))))
    log('VAL   MAE %.4f' % mv['mae'])
    log('epochs %d  sec/epoch %.2f  graph sec/epoch %.4f  A_edges %d  params %d'
        % (out['epochs_run'], out['sec_per_epoch'], out['graph_sec_per_epoch'],
           out['A_edges'], out['n_params']))
    pr = out.get('profile') or {}
    log('latency p50 %.2f ms  p95 %.2f ms  graph_peak %.1f MB  infer_peak %.1f MB  '
        'train_peak %.1f MB'
        % (pr.get('latency_p50_ms') or float('nan'),
           pr.get('latency_p95_ms') or float('nan'),
           pr.get('graph_peak_mb') or float('nan'),
           pr.get('infer_peak_mb') or float('nan'),
           out.get('peak_train_mb') or float('nan')))
    log('total wall %.1f s' % out['total_wall_sec'])
    return 0


def _cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == '__main__':
    sys.exit(main())
