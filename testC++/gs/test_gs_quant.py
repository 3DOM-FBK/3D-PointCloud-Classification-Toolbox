"""Measures the error of the `gs` v2 quantisation against the exact v1 values, for each source.

Prints the error report for SH at 8 and 16 bits, and the size of each option, so the bit depths can be chosen
from measurements. Exit code 1 if an error exceeds the budget in BUDGET (see below).

Usage: python test_gs_quant.py --data DIR
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'classifyViewer', 'viewer', 'utils_functions'))

import gs_format  # noqa: E402
import gs_io  # noqa: E402
import gs_quant  # noqa: E402

# Error budget for the default format (positions 24 bit, SH 8 bit). The limits are set before the
# measurements below, from what is invisible on screen at typical viewing distances:
# positions within 1 % of the splat's largest axis (99th percentile), rotation within 1e-3 rad,
# scale/opacity/colour within 1e-3 in their own units, SH rms error within 2 % of the SH rms signal.
BUDGET = {
    'pos_rel_splat_p99': 0.01,
    'rot_angle_max_rad': 1e-3,
    'scale_log_abs_max': 1e-3,
    'opacity_logit_abs_max': 1e-3,
    'dc_abs_max': 1e-3,
    'sh_rms_err_over_rms_signal': 0.02,
}


def chunks_for(n):
    return [{'id': i, 'start': s, 'count': min(gs_format.MAX_CHUNK, n - s)}
            for i, s in enumerate(range(0, n, gs_format.MAX_CHUNK))]


def roundtrip(gs, sh_bits):
    order = gs_format.morton_order(gs.xyz)
    chunks = chunks_for(len(gs))
    cols, chunks = gs_quant.encode(gs, order, chunks, sh_bits=sh_bits)
    arrays = {name: arr for name, (arr, _, _) in cols.items()}
    dec_rows = gs_quant.decode(arrays, chunks, gs.log_scale.shape[1])
    # decoded rows are in Morton order; put them back in source order
    inv = np.empty(len(gs), dtype=np.int64)
    inv[order] = np.arange(len(gs))
    dec = gs_io.GaussianSet(
        kind=gs.kind, source=gs.source, xyz=dec_rows['xyz'][inv], rot=dec_rows['rot'][inv],
        log_scale=dec_rows['log_scale'][inv], opacity=dec_rows['opacity'][inv], f_dc=dec_rows['f_dc'][inv],
        f_rest=dec_rows['f_rest'][inv] if gs.f_rest.shape[1] else gs.f_rest, sh_degree=gs.sh_degree)
    size = sum(arr.nbytes for arr in arrays.values()) + len(gs) * 4  # + id.bin
    return gs_quant.error_report(gs, dec), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    args = ap.parse_args()
    files = [('3dgs', '3DGS.ply'), ('2dgs', '2dgs_cardboard_point_cloud.ply'), ('fruit_sog', 'The Fruit Bowl.sog')]
    failures = []
    for name, fn in files:
        gs = gs_io.read_source(os.path.join(args.data, fn))
        n = len(gs)
        exact_bytes = n * (4 * (3 + 4 + gs.log_scale.shape[1] + 1 + 3 + gs.f_rest.shape[1]) + 4)
        print(f'\n== {name}: {n} splats, v1 size {exact_bytes / 2**20:.1f} MiB')
        for bits in (8, 16):
            rep, size = roundtrip(gs, bits)
            print(f'  sh {bits} bit: size {size / 2**20:.1f} MiB ({size / n:.1f} B/splat)')
            for k, v in rep.items():
                print(f'      {k:28s} {v:.6g}')
            if bits == 8:
                for k, lim in BUDGET.items():
                    if rep[k] > lim:
                        failures.append(f'{name}: {k} {rep[k]:.3g} > {lim}')
    print(f'\n{len(failures)} budget violation(s)')
    for f in failures:
        print('  -', f)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
