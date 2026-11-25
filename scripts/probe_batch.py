# -*- coding: utf-8 -*-
"""Extended max-runnable-batch probe (§7.4 / §7.6) for every method of one dataset.

Kept separate from profile_all.py so the heavy accuracy profiling does not have to be
repeated just to widen the batch ceiling.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from pbsg import engine, paths, profiling
from pbsg.model import GWNBackbone

POINTS = [('road', 32, 64, 0), ('dense', 32, 64, 0), ('topk', 32, 64, 4),
          ('pbsg', 32, 64, 0), ('pbsg', 32, 64, 4),
          ('fixed_cand', 32, 64, 0), ('rand_bud', 32, 64, 4)]
CANDS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--times', type=int, default=1,
                    help='repeat the whole sweep and keep the best, to survive a flaky OOM')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type != 'cuda':
        raise SystemExit('batch probe needs a GPU')
    d = paths.processed_dir(a.dataset)
    x, mask, mean, std, road, meta = engine.load_arrays(d)
    xn = ((x - mean) / std).astype(np.float32)
    tin, tout = meta['tin'], meta['tout']
    with open(d / 'meta.json') as f:
        sp = {k: np.asarray(v, dtype=np.int64)
              for k, v in json.load(f)['window_arrays'].items()}
    te = engine.WindowDS(xn, mask, sp['test'], tin, tout)

    out = {}
    for kind, K, M, m in POINTS:
        cand = d / ('cand_K%d_M%d.npz' % (K, M))
        cs_path = d / ('cand_K%d_M%d_stats.json' % (K, M))
        k_topk = None
        if cs_path.exists():
            with open(cs_path) as f:
                k_topk = int(round(json.load(f)['prior_support']['ci_size_mean'])) + m
        builder = engine.make_builder(kind, meta, road,
                                     cand if cand.exists() else None,
                                     K=K, M=M, m=m, d=16, seed=2026,
                                     k_topk=k_topk).to(dev)
        model = GWNBackbone(tin, tout).to(dev)
        best = None
        for _ in range(a.times):
            r = profiling.max_runnable_batch(model, builder, te, dev, cands=CANDS)
            if best is None or (r['max_batch_size'] or 0) > (best['max_batch_size'] or 0):
                best = r
        name = '%s_%s_K%d_M%d_m%d_s%d' % (a.dataset, kind, K, M, m, a.seed)
        out[name] = best
        d0 = os.path.join(paths.ROOT / '03_profile', name)
        os.makedirs(d0, exist_ok=True)
        with open(os.path.join(d0, 'batch.json'), 'w') as f:
            json.dump(best, f, indent=2)
        print('%-42s max_batch=%-6s %s' % (name, best['max_batch_size'],
                                           best['batch_probe']), flush=True)
        del model, builder
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
