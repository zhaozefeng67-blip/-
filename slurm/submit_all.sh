#!/bin/bash
# Submit every generated job wrapper that does not yet have a finished result.
# Jobs are submitted INDEPENDENTLY (no --dependency) so Slurm can pack as many as the
# account GRES limit allows.
cd $HOME/exp/pbsg || exit 1
n=0
for f in 05_slurm/jobs/*.slurm; do
  name=$(basename $f .slurm)
  if [ -f 03_runs/$name/result.json ]; then
    echo skip $name
    continue
  fi
  sbatch $f
  n=$((n+1))
done
echo submitted $n jobs
