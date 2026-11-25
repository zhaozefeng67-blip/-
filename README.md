# PBSG · Stage 2

Prior-Supported Budget-Constrained Sparse Graph construction for traffic forecasting.
This repository is the **Stage-2 experiment code** (parameter and structure validation),
plus the measurement harness for the plan's §7.2–7.4 reporting requirements.

The research question is not "build a better STGNN". A single frozen Graph-WaveNet-style
backbone is shared by every method, and only the graph builder is swapped:

```
Road prior -> Support (bounded candidate set C_i)
           -> Weight  (learn scores only on candidate edges)
           -> Budget  (at most m prior-external edges per node from a bounded pool F_i)
           -> sparse adjacency A -> fixed backbone -> forecast
```

## Layout

```
pbsg/            package
  paths.py       project-relative directories
  dataio.py      raw parquet -> arrays, split, train-only scaler, road-graph audit
  graphop.py     CSR road graph, C_i (bounded BFS), F_i (bounded rejection sampling)
  builders.py    graph builders A-G, one sparse/dense operator out
  model.py       frozen GWN-style backbone (only A is injected)
  engine.py      window dataset, masked metrics, training loop, inference profiling
  profiling.py   §7.2 structure, §7.3 timing, §7.4 decomposition / throughput / batch probe
scripts/         CLIs (see below)
slurm/           job templates and submission helpers
docs/            protocol, results, §7.2-7.4 measurement write-up, aligned report
figs/            Pareto figures produced by the accuracy sweep
```

## Methods

| id | name | support | weights |
|----|------|---------|---------|
| A | Road-Fixed | released road graph | fixed, row-normalised |
| B | Dense-Adaptive | all $N(N-1)$ pairs | learned $s_{ij}=\mathrm{ReLU}(e_i^\top e_j)$ |
| C | Global-TopK | top-$k$ of all pairs, $k=\mathrm{round}(\overline{\lvert C_i\rvert})+m$ | learned |
| D | PBSG-$m{=}0$ | $C_i$ only | learned |
| E | PBSG-$m{>}0$ | $C_i$ plus top-$m$ of $F_i$ | learned |
| F | Fixed-Candidate | same support as D | **fixed** (isolates the value of learning) |
| G | Random-Budget | $C_i$ plus $m$ random from $F_i$ (fixed seed) | learned (isolates the value of selecting) |

Global-TopK's $k$ is aligned to PBSG's average out-degree on purpose: the honest
comparison is at equal average degree, not at the nominal $K+m$.

## Data

PeMS flow and the released sparse road edge lists come from two Hugging Face mirrors of
the original releases (the BNBU cluster cannot reach huggingface.co directly, but can
reach `hf-mirror.com`):

- `Davidham3/pems-stsgcn` — PEMS03/04/07/08 `_series.parquet` + `_edges.parquet`.
  PEMS03 additionally ships `_sensors.parquet`, which is the only way to map its
  Caltrans station ids onto the series axis; PEMS04/08 edge ids already index the axis.
- `jimmygao3218/PEMS07` — Graph-WaveNet-format `PEMS07.npz` + `adj`/`distance` pickles,
  whose row order *is* the series node axis. Used for PEMS07 because the STSGCN release
  keeps raw station ids there with no sensors table. Cross-check: its adjacency has
  1732 = 866 x 2 edges, exactly the symmetrised STSGCN road edge list.

Datasets are **not** vendored here. Put the raw files under `02_data/raw/` (see
`pbsg/paths.py` for the root) and run the prepare step.

## Quick start

```bash
export PYTHONPATH=$HOME/exp/pbsg/pysite:$HOME/exp/pbsg/01_code

# 1) arrays, split, scaler, road graph   (per dataset)
python scripts/prepare.py --dataset PEMS08

# 2) offline candidate caches            (per K,M; seven (K,M) points cover §8.2)
for km in 8,64 16,64 32,64 64,64 32,16 32,32 32,128; do
  K=${km%,*}; M=${km#*,}
  python scripts/build_candidates.py --dataset PEMS08 --K $K --M $M
done

# 3) one accuracy run
python scripts/run_one.py --dataset PEMS08 --kind pbsg --K 32 --M 64 --m 4 --seed 42

# 4) a sweep: generate one sbatch wrapper per run, then submit them independently
python scripts/gen_matrix.py --datasets PEMS08 --tier full
bash slurm/submit_all.sh

# 5) tables and figures
python scripts/aggregate.py --csv figs/stage2_accuracy.csv
python scripts/plot_pareto.py

# 6) the measurement pass (§7.2-7.4)
python scripts/gen_profile_jobs.py --datasets PEMS08 PEMS04 PEMS03 PEMS07
bash slurm/submit_profile.sh
python scripts/aggregate_profile.py
```

`scripts/gen_matrix.py --tier` selects the run set: `core` (A-G at one point),
`focus` (budget and pool sweeps only), `full` (core plus the K/M/m grids).

## Environment notes

Target host is a Slurm cluster with RTX 4090 nodes. Two things that cost time to learn:

- **Do not chain runs with `--dependency`.** Submit every run independently so the
  scheduler can keep as many tasks running as the account quota allows. A dependency
  chain serialises everything and wastes hours.
- **Only sm_89 is usable** with the shared torch build (`2.7.0+cu126`): request
  `--gres=gpu:4090:1` and exclude the node with the broken CUDA stack. Both job
  templates here do this and self-check `torch.cuda.is_available()` before running.
- V100 nodes are **not** usable with that wheel (sm_70 is not in the build).

`requirements.txt` lists the Python side. `pyarrow` is only needed for the parquet
sources and is installed with `pip install --target <dir> pyarrow` so the shared venv
stays untouched.

## Findings (short)

Full numbers and caveats are in `docs/`. The three that matter:

1. **The budget mechanism pays off exactly where the prior support is weakest.**
   Gain from $m{=}0$ to $m{=}4$ is $+0.03$ / $-1.05$ / $-1.28$ MAE on PEMS08 / PEMS04 /
   PEMS07, tracking $\overline{\lvert C_i\rvert}$ = 24.18 / 12.41 / 8.01.
2. **Diminishing returns start at $m\approx2$**, not gradually: PEMS07 gains all of
   $-1.30$ at $m{=}2$ and gets worse again at $m{=}8$.
3. **The wall-clock benefit does not exist at this scale.** PBSG evaluates 35-42x fewer
   node pairs than dense and stores 13.9x fewer bytes at $N{=}883$, yet runs 1.5-1.7x
   slower per epoch because sparse kernels are slower than dense at these sizes.
   The defensible advantage is the number of scored pairs and adjacency storage.

Also stated plainly in the docs: **PBSG does not beat Global-TopK on accuracy** at equal
average degree (17.842 vs 17.907 / 22.036 vs 21.917 / 24.101 vs 24.190).
