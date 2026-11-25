# -*- coding: utf-8 -*-
"""Emit one sbatch wrapper per (dataset, method) for the §7.2-7.5 profiling pass."""
import argparse
import csv
import os
import stat

from pbsg import paths

TMPL = """#!/bin/bash
#SBATCH --job-name=pbsgprof_{name}
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
exec $HOME/exp/desub/venv/bin/python 01_code/scripts/profile_all.py \\
  --dataset {dataset} --kind {kind} --K {K} --M {M} --m {m} --seed {seed} \\
  --reps {reps} --batch-size {bs} --workers 4
"""

POINTS = [('road', 32, 64, 0), ('dense', 32, 64, 0), ('topk', 32, 64, 4),
          ('pbsg', 32, 64, 0), ('pbsg', 32, 64, 4),
          ('fixed_cand', 32, 64, 0), ('rand_bud', 32, 64, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['PEMS08', 'PEMS04', 'PEMS03', 'PEMS07'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--timelimit', default='1:00:00')
    a = ap.parse_args()

    jobs = paths.ROOT / '05_slurm' / 'profile_jobs'
    os.makedirs(jobs, exist_ok=True)
    logs = os.path.expanduser('~/pbsg_logs')
    rows = []
    for ds in a.datasets:
        for kind, K, M, m in POINTS:
            name = '%s_%s_K%d_M%d_m%d_s%d' % (ds, kind, K, M, m, a.seed)
            body = TMPL.format(name=name, dataset=ds, kind=kind, K=K, M=M, m=m,
                               seed=a.seed, reps=a.reps, bs=a.batch_size,
                               timelimit=a.timelimit, logs=logs)
            fp = jobs / (name + '.slurm')
            fp.write_text(body)
            fp.chmod(fp.stat().st_mode | stat.S_IEXEC)
            rows.append({'name': name, 'dataset': ds, 'kind': kind, 'slurm': str(fp)})

    csv_path = paths.ROOT / '05_slurm' / 'profile_matrix.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('datasets %s' % a.datasets)
    print('methods  %d' % len(POINTS))
    print('jobs     %d' % len(rows))
    print('wrappers %s' % jobs)
    print('submit:  bash 05_slurm/submit_profile.sh')


if __name__ == '__main__':
    main()
