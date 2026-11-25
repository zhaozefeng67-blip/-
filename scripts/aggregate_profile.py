# -*- coding: utf-8 -*-
"""Collect the profiling pass into the §7.2 / §7.3 / §7.4 / §7.5 tables."""
import argparse
import csv
import glob
import json
import os

from pbsg import paths

DS_ORDER = ['PEMS08', 'PEMS04', 'PEMS03', 'PEMS07']
METH_ORDER = ['road', 'dense', 'topk', 'pbsg', 'fixed_cand', 'rand_bud']
LABEL = {'road': 'A Road-Fixed', 'dense': 'B Dense-Adaptive', 'topk': 'C Global-TopK',
         'fixed_cand': 'F Fixed-Candidate', 'rand_bud': 'G Random-Budget'}


def mlabel(r):
    if r['kind'] == 'pbsg':
        return 'D PBSG-m0' if r['m'] == 0 else 'E PBSG-m%d' % r['m']
    return LABEL.get(r['kind'], r['kind'])


def kb(b):
    return '%.1f' % (float(b) / 1024.0) if b is not None else '-'


def num(v, fmt='%.3f'):
    return fmt % v if v is not None else '-'


def load_profiles(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, '*', 'profile.json'))):
        base = os.path.dirname(p)
        if os.path.basename(base).startswith('_'):
            continue                      # skip smoke / scratch runs
        with open(p) as f:
            r = json.load(f)
        # an extended batch probe (probe_batch.py) supersedes the in-pass ceiling
        bp = os.path.join(base, 'batch.json')
        if os.path.exists(bp):
            with open(bp) as f:
                r['batch'] = json.load(f)
                r['batch']['source'] = 'extended probe'
        rows.append(r)
    return rows


def load_cand_stats(dataset, K, M):
    """Single source of truth for the offline candidate cost, read from the cache dir."""
    p = paths.PROC / dataset / ('cand_K%d_M%d_stats.json' % (K, M))
    if not p.exists():
        return {}
    with open(p) as f:
        st = json.load(f)
    tm = st.get('timing', {})
    stg = st.get('storage', {})
    ps = st.get('prior_support', {})
    return {'prior_cand_gen_sec': tm.get('prior_cand_gen_sec'),
            'external_pool_gen_sec': tm.get('external_pool_gen_sec'),
            'k_eff': ps.get('K_eff'),
            'ci_size_mean': ps.get('ci_size_mean'),
            'prior_coverage': st.get('prior_coverage', {}).get('coverage'),
            'candidate_storage_bytes': stg.get('cand_bytes'),
            'pool_storage_bytes': stg.get('pool_bytes'),
            'cache_file_bytes': stg.get('cache_file_bytes')}


def load_train_epoch(root):
    """sec/epoch from the accuracy sweep, when that run exists."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, '*', 'result.json'))):
        with open(p) as f:
            r = json.load(f)
        c = r['config']
        out[(c['dataset'], c['kind'], c['K'], c['M'], c['m'])] = r['sec_per_epoch']
    return out


def key(r):
    return (r['dataset'], r['kind'], r['K'], r['M'], r['m'])


def table(headers, rows, aligns):
    import unicodedata

    def dw(s):
        return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)

    def pad(s, w, right=False):
        g = max(0, w - dw(s))
        return (' ' * g + s) if right else (s + ' ' * g)

    widths = [dw(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], dw(c))
    out = ['| ' + ' | '.join(pad(h, widths[i]) for i, h in enumerate(headers)) + ' |']
    seps = [(':' + '-' * (widths[i] - 1)) if aligns[i] == 'l' else ('-' * (widths[i] - 1) + ':')
            for i in range(len(headers))]
    out.append('| ' + ' | '.join(seps) + ' |')
    for r in rows:
        out.append('| ' + ' | '.join(pad(c, widths[i], aligns[i] == 'r')
                                     for i, c in enumerate(r)) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profiles', default=str(paths.ROOT / '03_profile'))
    ap.add_argument('--runs', default=str(paths.RUNS))
    ap.add_argument('--csv-prefix', default=str(paths.FIGS / 'profile'))
    a = ap.parse_args()

    profs = load_profiles(a.profiles)
    if not profs:
        raise SystemExit('no profile.json under %s' % a.profiles)
    tr_ep = load_train_epoch(a.runs)
    order = {d: i for i, d in enumerate(DS_ORDER)}

    def sk(r):
        return (order.get(r['dataset'], 99), METH_ORDER.index(r['kind'])
                if r['kind'] in METH_ORDER else 9, r['m'])

    profs.sort(key=sk)

    # ---------------- 7.2 ----------------
    t72 = []
    for r in profs:
        s = r['structure']
        o = load_cand_stats(r['dataset'], r['K'], r['M'])
        edges_ns = s['edges_without_self']
        t72.append([r['dataset'], mlabel(r),
                    '{:,}'.format(edges_ns),
                    '{:,}'.format(s['edges_with_self']),
                    '%.2f' % s['avg_out_degree'],
                    str(s['max_out_degree']),
                    '%.2e' % s['graph_density'],
                    '{:,}'.format(s['scored_pairs']),
                    '{:,}'.format(s['prior_candidate_count']),
                    '{:,}'.format(s['budget_edge_count']),
                    '%.4f' % s['budget_edge_ratio'],
                    num(o.get('prior_coverage'), '%.4f'),
                    kb(s['adjacency_bytes']),
                    kb(edges_ns * 20.0),
                    kb(o.get('cache_file_bytes'))])
    H72 = ['数据集', '方法', '边数(不含自环)', '边数(含自环)', '平均出度', '最大出度',
           '图密度', '打分数', '候选边数', '预算边数', '预算边占比', '先验覆盖率',
           '实际邻接 KB', '稀疏等价 KB', '候选缓存 KB']
    A72 = ['l', 'l'] + ['r'] * 13

    # ---------------- 7.3 ----------------
    t73 = []
    for r in profs:
        b, i = r['builder_timing'], r['inference']
        o = load_cand_stats(r['dataset'], r['K'], r['M'])
        pg = o.get('prior_cand_gen_sec')
        eg = o.get('external_pool_gen_sec')
        t73.append([r['dataset'], mlabel(r),
                    num(pg * 1000.0 if pg is not None else None, '%.1f'),
                    num(eg * 1000.0 if eg is not None else None, '%.1f'),
                    '%.3f' % b['graph_ms_median'],
                    '%.3f' % b['graph_ms_p95'],
                    num(tr_ep.get(key(r)), '%.2f'),
                    '%.2f' % i['infer_cached_ms_median'],
                    '%.2f' % i['infer_build_ms_median'],
                    '%.2f' % i['infer_build_ms_p95']])
    H73 = ['数据集', '方法', 'C_i 生成 ms', 'F_i 生成 ms', '构图 median ms', '构图 p95 ms',
           's/epoch', '推理 cached p50 ms', '推理 build p50 ms', '推理 build p95 ms']
    A73 = ['l', 'l'] + ['r'] * 8

    # ---------------- 7.4 / 7.5 ----------------
    t74 = []
    for r in profs:
        e = r['epoch']
        bp = r['batch']
        step_ms = (1000.0 * e['decomp_total_sec'] / e['decomp_steps']
                   if e['decomp_steps'] else None)
        t74.append([r['dataset'], mlabel(r),
                    num(tr_ep.get(key(r)), '%.2f'),
                    num(step_ms, '%.1f'),
                    '%.1f' % e['throughput_samples_per_sec']
                    if e['throughput_samples_per_sec'] else '-',
                    '{:,}'.format(r['n_params']),
                    str(bp['max_batch_size']) if bp['max_batch_size'] else '-',
                    '%.1f' % e['decomp_graph_pct'],
                    '%.1f' % e['decomp_fwd_pct'],
                    '%.1f' % e['decomp_bwd_pct'],
                    '%.1f' % e['decomp_opt_pct'],
                    '%.1f' % e['decomp_data_pct'],
                    num(e['peak_train_mb'], '%.0f')])
    H74 = ['数据集', '方法', 's/epoch', '插桩 step ms', '吞吐 samples/s', '参数量',
           '最大 batch', 'graph %', 'fwd %', 'bwd %', 'opt %', 'data %', '峰值显存 MB']
    A74 = ['l', 'l'] + ['r'] * 11

    for name, H, rows, A in (('7.2', H72, t72, A72), ('7.3', H73, t73, A73),
                             ('7.4', H74, t74, A74)):
        print()
        print('=' * 78)
        print('SECTION %s' % name)
        print('=' * 78)
        print(table(H, rows, A))

    for name, H, rows in (('7.2', H72, t72), ('7.3', H73, t73), ('7.4', H74, t74)):
        fp = '%s_%s.csv' % (a.csv_prefix, name.replace('.', ''))
        with open(fp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(H)
            w.writerows(rows)
        print('wrote', fp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
