# -*- coding: utf-8 -*-
"""D1 CLI: build processed arrays for one dataset."""
import argparse
import json
import sys

from pbsg import dataio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--tin', type=int, default=12)
    ap.add_argument('--tout', type=int, default=12)
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.6, 0.2, 0.2])
    ap.add_argument('--zero-as-missing', action='store_true')
    ap.add_argument('--no-symmetrize', action='store_true')
    a = ap.parse_args()

    meta = dataio.build(a.dataset, a.tin, a.tout, tuple(a.ratios), a.zero_as_missing,
                        symmetrize=not a.no_symmetrize)

    sa, ea, na = meta['series_audit'], meta['edge_audit'], meta['node_id_audit']
    b, af = ea['before'], ea['after']
    print('=' * 78)
    print('%s  T=%d N=%d  tin/tout=%d/%d  ratios=%s'
          % (meta['dataset'], meta['T'], meta['N'], meta['tin'], meta['tout'], meta['ratios']))
    print('-' * 78)
    print('split bounds (original time axis): %s' % json.dumps(meta['split_bounds']))
    print('windows inside each split      : %s' % json.dumps(meta['windows']))
    print('-' * 78)
    print('[series] zero_frac=%.5f  node zero-frac min/med/max = %.4f / %.4f / %.4f'
          % (sa['zero_frac_all'], sa['node_zero_frac_min'],
             sa['node_zero_frac_median'], sa['node_zero_frac_max']))
    print('[series] nodes with >50%% zeros = %d   value range [%.1f, %.1f]   zero_as_missing=%s'
          % (sa['nodes_zero_frac_gt_50pct'], sa['value_min'], sa['value_max'],
             meta['zero_as_missing']))
    print('[edges ] symmetrized=%s  rows %d -> %d (dedupe removed %d, self-loops %d)'
          % (ea['symmetrized'], ea['rows_as_released'], ea['rows_final'],
             ea['duplicate_rows_removed'], b['self_loops']))
    print('[edges ] as released (directed): avg_out=%.3f max_out=%d  out_deg0=%d in_deg0=%d either0=%d'
          % (b['avg_out_degree'], b['max_out_degree'],
             b['out_degree_zero'], b['in_degree_zero'], b['either_degree_zero']))
    print('[edges ] used for C_i          : avg_out=%.3f max_out=%d  isolated=%d'
          % (af['avg_out_degree'], af['max_out_degree'], af['either_degree_zero']))
    print('[edges ] cost unit guess=%s  median=%.3f  range=[%.3f, %.3f]'
          % (ea['cost_unit'], ea['cost_median'], ea['cost_min'], ea['cost_max']))
    print('[nodeid] edge ids span [%d, %d]  nodes_used=%d  nodes_without_edges=%d'
          % (na['edge_id_min'], na['edge_id_max'],
             na['unique_edge_nodes'], na['nodes_without_edges']))
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
