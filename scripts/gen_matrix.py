# -*- coding: utf-8 -*-
"""Generate the Stage-2 run matrix and one sbatch wrapper per run.

Design choices that matter for wall clock:

* every run is submitted as an INDEPENDENT job, never a ``--dependency`` chain, so Slurm
  can keep as many tasks running as the account GRES limit allows;
* ``--gres=gpu:4090:1`` constrains placement to sm_89 nodes, which is what the shared
  venv's torch +cu126 build supports, and ``--exclude=hpcgpu03`` skips the node with the
  broken CUDA stack;
* tiers let the fast datasets go first: ``core`` is the minimum defensible Stage-2 set,
  ``full`` adds the K / M grids.
"""
import argparse
import csv
import os
import stat

from pbsg import paths

SLURM_TMPL = """#!/bin/bash
#SBATCH --job-name=pbsg_{name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --exclude=hpcgpu03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={timelimit}
#SBATCH --output={logs}/{name}.out
#SBATCH --error={logs}/{name}.err
set -uo pipefail
export PYTHONPATH=$HOME/exp/pbsg/pysite:$HOME/exp/pbsg/01_code
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
cd $HOME/exp/pbsg
$HOME/exp/desub/venv/bin/python - <<'PY' || exit 3
import sys, torch
ok = torch.cuda.is_available()
print('cuda_available', ok, torch.cuda.get_device_name(0) if ok else '-')
sys.exit(0 if ok else 3)
PY
exec $HOME/exp/desub/venv/bin/python 01_code/scripts/run_one.py \\
  --dataset {dataset} --kind {kind} --K {K} --M {M} --m {m} --seed {seed} \\
  --max-epochs {epochs} --patience {patience} --batch-size {bs} --workers 4
"""


def core_points():
    """Minimum defensible Stage-2 set: A-E plus the two structural controls."""
    p = [('road', 32, 64, 0), ('dense', 32, 64, 0), ('topk', 32, 64, 4),
         ('pbsg', 32, 64, 0), ('pbsg', 32, 64, 4),
         ('fixed_cand', 32, 64, 0), ('rand_bud', 32, 64, 4)]
    return p


def budget_points():
    return [('pbsg', 32, 64, 2), ('pbsg', 32, 64, 8), ('pbsg', 32, 64, 16)]


def k_points():
    return [('pbsg', 8, 64, 8), ('pbsg', 16, 64, 8), ('pbsg', 64, 64, 8)]


def m_points():
    return [('pbsg', 32, 16, 8), ('pbsg', 32, 32, 8), ('pbsg', 32, 128, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['PEMS08'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--tier', choices=['core', 'focus', 'full'], default='core')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--timelimit', default='2:00:00')
    a = ap.parse_args()

    pts = core_points()
    if a.tier == 'focus':
        pts = budget_points() + m_points()
    elif a.tier == 'full':
        pts = pts + budget_points() + k_points() + m_points()

    seen, dedup = set(), []
    for kind, K, M, m in pts:
        key = (kind, K, M, m)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)

    jobs_dir = paths.ROOT / '05_slurm' / 'jobs'
    os.makedirs(jobs_dir, exist_ok=True)
    logs = os.path.expanduser('~/pbsg_logs')
    os.makedirs(os.path.expanduser(os.path.join('~', 'pbsg_logs')), exist_ok=True)

    rows = []
    for ds in a.datasets:
        for i, (kind, K, M, m) in enumerate(dedup):
            for seed in a.seeds:
                name = '%s_%s_K%d_M%d_m%d_s%d' % (ds, kind, K, M, m, seed)
                body = SLURM_TMPL.format(
                    name=name, dataset=ds, kind=kind, K=K, M=M, m=m, seed=seed,
                    epochs=a.epochs, patience=a.patience, bs=a.batch_size,
                    timelimit=a.timelimit, logs=logs)
                fp = jobs_dir / (name + '.slurm')
                fp.write_text(body)
                fp.chmod(fp.stat().st_mode | stat.S_IEXEC)
                rows.append({'name': name, 'dataset': ds, 'kind': kind, 'K': K,
                             'M': M, 'm': m, 'seed': seed, 'slurm': str(fp)})

    csv_path = paths.ROOT / '05_slurm' / ('matrix_%s_%s.csv' % ('-'.join(a.datasets), a.tier))
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('configs per dataset : %d' % len(dedup))
    print('seeds               : %s' % a.seeds)
    print('datasets            : %s' % a.datasets)
    print('total runs          : %d' % len(rows))
    print('matrix csv          : %s' % csv_path)
    print('job wrappers        : %s (%d files)' % (jobs_dir, len(rows)))
    print()
    print('submit with:  for f in %s/jobs/*.slurm; do sbatch "$f"; done'
          % (paths.ROOT / '05_slurm'))


if __name__ == '__main__':
    main()
