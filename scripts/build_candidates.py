# -*- coding: utf-8 -*-
"""D2 CLI: build + cache C_i (road prior support) and F_i (prior-external pool)."""
import argparse
import json
import sys
import time

import numpy as np

from pbsg import graphop, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--max-hops', type=int, default=4)
    a = ap.parse_args()

    d = paths.processed_dir(a.dataset)
    meta_path = d / 'meta.json'
    if not meta_path.exists():
        raise SystemExit('run prepare.py for %s first' % a.dataset)
    with open(meta_path) as f:
        meta = json.load(f)
    N = meta['N']

    road = np.load(d / 'road.npz')
    indptr, indices = graphop.csr_from_edges(road['src'], road['dst'], N)

    t0 = time.perf_counter()
    C, K_eff, cstats = graphop.build_prior_candidates(indptr, indices, a.K, a.max_hops)
    t_prior = time.perf_counter() - t0

    t0 = time.perf_counter()
    F, Fn, fstats = graphop.build_external_pool(N, C, a.M, a.seed)
    t_pool = time.perf_counter() - t0

    cov = graphop.prior_coverage(C, road['src'], road['dst'])

    stats = {
        'dataset': a.dataset,
        'N': int(N),
        'K': a.K,
        'K_eff': int(K_eff),
        'M': a.M,
        'seed': a.seed,
        'prior_support': cstats,
        'external_pool': fstats,
        'prior_coverage': cov,
        # 7.3 items 1 and 2: the offline candidate-generation costs, timed separately
        'timing': {'prior_cand_gen_sec': float(t_prior),
                   'external_pool_gen_sec': float(t_pool),
                   'total_gen_sec': float(t_prior + t_pool)},
    }
    graphop.save_candidates(d / ('cand_K%d_M%d.npz' % (a.K, a.M)), C, F, stats)

    print('=' * 72)
    print('%s  N=%d  K=%d -> K_eff=%d   M=%d   seed=%d'
          % (a.dataset, N, a.K, K_eff, a.M, a.seed))
    print('-' * 72)
    print('[prior] |C_i| min/mean/max = %d / %.2f / %d   total candidate edges = %d'
          % (cstats['ci_size_min'], cstats['ci_size_mean'],
             cstats['ci_size_max'], cstats['ci_edges_total']))
    print('[prior] nodes whose 1-hop road degree > K = %d (kept in full, K_eff grew)'
          % cstats['nodes_with_1hop_over_K'])
    print('[prior] hop composition: %s' % json.dumps(cstats['hop_composition']))
    print('[prior] road-edge coverage by C_i = %d/%d (%.4f)'
          % (cov['covered'], cov['road_edges'], cov['coverage']))
    print('[pool ] |F_i| min/mean/max = %d / %.2f / %d   total = %d'
          % (fstats['fi_size_min'], fstats['fi_size_mean'],
             fstats['fi_size_max'], fstats['fi_edges_total']))
    print('[pool ] nodes not filled to M = %d' % fstats['nodes_not_filled'])
    print('[storage] C %.1f KB   F %.1f KB'
          % (C.nbytes / 1024.0, F.nbytes / 1024.0))
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
