#!/bin/bash
# Submit profiling jobs that have no finished profile.json yet.
cd $HOME/exp/pbsg || exit 1
n=0
for f in 05_slurm/profile_jobs/*.slurm; do
  name=$(basename $f .slurm)
  if [ -f 03_profile/$name/profile.json ]; then
    echo skip $name
    continue
  fi
  sbatch $f
  n=$((n+1))
done
echo submitted $n jobs
