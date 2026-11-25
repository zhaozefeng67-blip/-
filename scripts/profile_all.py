# -*- coding: utf-8 -*-
"""Full §7.2 / §7.3 / §7.4 / §7.5 measurement pass for one (dataset, method, config)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pbsg import engine, paths, profiling
from pbsg.model import GWNBackbone


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
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--dilations', type=int, nargs='+', default=[1, 2, 4, 4])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--decomp-epochs', type=int, default=3)
    ap.add_argument('--skip-batch-probe', action='store_true')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = paths.processed_dir(a.dataset)
    cand_path = d / ('cand_K%d_M%d.npz' % (a.K, a.M))
    cand_stats_path = d / ('cand_K%d_M%d_stats.json' % (a.K, a.M))
    cand_stats = None
    if cand_stats_path.exists():
        with open(cand_stats_path) as f:
            cand_stats = json.load(f)
    if a.kind in ('pbsg', 'fixed_cand', 'rand_bud') and not cand_path.exists():
        raise SystemExit('missing candidate cache: %s' % cand_path)

    x, mask, mean, std, road, meta = engine.load_arrays(d)
    xn = ((x - mean) / std).astype(np.float32)
    tin, tout = meta['tin'], meta['tout']
    with open(d / 'meta.json') as f:
        sp = {k: np.asarray(v, dtype=np.int64)
              for k, v in json.load(f)['window_arrays'].items()}

    tr = engine.WindowDS(xn, mask, sp['train'], tin, tout)
    te = engine.WindowDS(xn, mask, sp['test'], tin, tout)
    dl_tr = DataLoader(tr, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                       drop_last=True, pin_memory=(dev.type == 'cuda'))

    k_topk = None
    if a.kind == 'topk' and cand_stats is not None:
        k_topk = int(round(cand_stats['prior_support']['ci_size_mean'])) + a.m

    builder = engine.make_builder(a.kind, meta, road, cand_path if cand_path.exists() else None,
                                  K=a.K, M=a.M, m=a.m, d=a.d_emb, seed=a.cand_seed,
                                  k_topk=k_topk).to(dev)
    model = GWNBackbone(tin, tout, channels=a.channels,
                        dilations=tuple(a.dilations)).to(dev)

    name = '%s_%s_K%d_M%d_m%d_s%d' % (a.dataset, a.kind, a.K, a.M, a.m, a.seed)
    outdir = a.outdir or (paths.ROOT / '03_profile' / name)
    os.makedirs(outdir, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with open(os.path.join(outdir, 'profile.log'), 'a') as f:
            f.write(m + '\n')

    log('=' * 68)
    log('profile %s  device=%s  reps=%d' % (name, dev.type, a.reps))
    log('=' * 68)

    res = {'name': name, 'dataset': a.dataset, 'kind': a.kind, 'K': a.K, 'M': a.M,
           'm': a.m, 'seed': a.seed, 'device': str(dev), 'N': int(meta['N']),
           'T': int(meta['T']), 'batch_size': a.batch_size,
           'dilations': a.dilations, 'channels': a.channels,
           'n_params': int(sum(p.numel() for p in model.parameters())
                           + (builder.emb.numel() if builder.emb is not None else 0))}

    # ---- 7.2 structure (built once, full graph) ----
    model.eval()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        A = builder()
    res['structure'] = profiling.structural_stats(builder, A, meta['N'], cand_stats)
    del A
    log('[7.2] edges excl self %d  avg_out %.3f  max_out %d  density %.3e'
        % (res['structure']['edges_without_self'], res['structure']['avg_out_degree'],
           res['structure']['max_out_degree'], res['structure']['graph_density']))
    log('[7.2] prior %d  budget %d  ratio %.4f  adj_bytes %d'
        % (res['structure']['prior_candidate_count'], res['structure']['budget_edge_count'],
           res['structure']['budget_edge_ratio'], res['structure']['adjacency_bytes']))

    # ---- 7.3 graph builder alone ----
    res['builder_timing'] = profiling.time_builder(builder, dev, a.warmup, a.reps)
    bt = res['builder_timing']
    log('[7.3] builder mean %.3f ms  median %.3f  p95 %.3f  peak %.1f MB'
        % (bt['graph_ms_mean'], bt['graph_ms_median'], bt['graph_ms_p95'],
           bt['graph_builder_peak_mb'] or -1))

    # ---- 7.3/7.4 inference latency ----
    x_one = te[0][0].unsqueeze(0).to(dev)
    res['inference'] = profiling.time_inference(model, builder, x_one, dev,
                                                a.warmup, a.reps)
    inf = res['inference']
    log('[7.3] infer cached p50 %.2f ms / build-incl p50 %.2f ms'
        % (inf['infer_cached_ms_median'], inf['infer_build_ms_median']))

    # ---- 7.4 epoch decomposition ----
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    res['epoch'] = profiling.epoch_decomposition(model, builder, dl_tr, dev,
                                                 a.decomp_epochs)
    res['epoch']['peak_train_mb'] = profiling.peak_train_memory(dev)
    e = res['epoch']
    log('[7.4] decomp steps %d  data %.1f%%  graph %.1f%%  fwd %.1f%%  bwd %.1f%%  opt %.1f%%'
        % (e['decomp_steps'], e['decomp_data_pct'], e['decomp_graph_pct'],
           e['decomp_fwd_pct'], e['decomp_bwd_pct'], e['decomp_opt_pct']))
    log('[7.4] throughput %.1f samples/s  peak train %.1f MB'
        % (e['throughput_samples_per_sec'] or -1, e['peak_train_mb'] or -1))

    # ---- 7.4 max runnable batch ----
    if a.skip_batch_probe:
        res['batch'] = {'max_batch_size': None, 'batch_probe': {}, 'skipped': True}
    else:
        res['batch'] = profiling.max_runnable_batch(model, builder, te, dev)
    log('[7.4] max batch %s  probe %s'
        % (res['batch']['max_batch_size'], res['batch']['batch_probe']))

    # ---- 7.3 offline candidate generation (copied from the cache stats) ----
    if cand_stats:
        res['offline'] = {
            'prior_cand_gen_sec': cand_stats.get('timing', {}).get('prior_cand_gen_sec'),
            'external_pool_gen_sec': cand_stats.get('timing', {}).get('external_pool_gen_sec'),
            'k_eff': cand_stats['prior_support']['K_eff'],
            'ci_size_mean': cand_stats['prior_support']['ci_size_mean'],
            'prior_coverage': cand_stats['prior_coverage']['coverage'],
            'candidate_storage_bytes': cand_stats.get('storage', {}).get('cand_bytes'),
            'pool_storage_bytes': cand_stats.get('storage', {}).get('pool_bytes'),
            'cache_file_bytes': cand_stats.get('storage', {}).get('cache_file_bytes'),
        }
    else:
        res['offline'] = {}

    with open(os.path.join(outdir, 'profile.json'), 'w') as f:
        json.dump(res, f, indent=2)
    log('wrote %s' % os.path.join(outdir, 'profile.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
