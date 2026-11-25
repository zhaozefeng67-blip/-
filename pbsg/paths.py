# -*- coding: utf-8 -*-
"""Project paths. Code lives in ``<proj>/01_code/pbsg`` so the project root is parents[2]."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / '02_data' / 'raw'
PROC = ROOT / '02_data' / 'processed'
RUNS = ROOT / '03_runs'
FIGS = ROOT / '04_figs'
CONF = ROOT / '01_code' / 'configs'


def ensure_dirs():
    for d in (PROC, RUNS, FIGS, CONF):
        os.makedirs(d, exist_ok=True)


def processed_dir(dataset):
    d = PROC / dataset
    os.makedirs(d, exist_ok=True)
    return d
