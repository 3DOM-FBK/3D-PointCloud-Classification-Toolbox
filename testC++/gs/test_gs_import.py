"""Checks for the Gaussian splat importers and the `gs` format.

For each source (3DGS PLY, 2DGS PLY, SOG v2):
  1. read the source with gs_io;
  2. write the `gs` v1 folder (float32), read it back in source order, compare every column (bit-exact);
  3. write the `gs` v2 folder (quantised), read it back, check the error report against test_gs_quant.BUDGET;
  4. write a PLY from the v2 data, read it back, compare every field (bit-exact, rot after normalisation);
  5. compare the source with the reference output of PlayCanvas splat-transform 3.9.0 (`--ref-dir`), when given.

Usage (inside the app image, from the repo):
  python test_gs_import.py --data DIR --out DIR [--ref-dir DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'classifyViewer', 'viewer', 'utils_functions'))

import gs_format  # noqa: E402
import gs_io  # noqa: E402
import gs_quant  # noqa: E402
import test_gs_quant  # noqa: E402  (budget, same folder)

FIELDS = ['xyz', 'rot', 'log_scale', 'opacity', 'f_dc', 'f_rest']

failures = []


def check(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        failures.append(msg)


def compare(a, b, label, tol=0.0, loose=()):
    """Compare two GaussianSets field by field. tol=0 means bit-exact, except the `loose` fields
    which are compared within 1e-6. Returns the max abs difference per field."""
    diffs = {}
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if x.shape != y.shape:
            check(False, f'{label}: {f} shape {x.shape} vs {y.shape}')
            continue
        if x.size == 0:
            diffs[f] = 0.0
            continue
        d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        diffs[f] = d
        if tol == 0.0 and f not in loose:
            check(np.array_equal(x, y, equal_nan=True), f'{label}: {f} bit-exact (max diff {d:g})')
        else:
            limit = tol if tol else 1e-6
            check(d <= limit, f'{label}: {f} max diff {d:g} <= {limit:g}')
    return diffs


def run_source(name, path, out_dir, ref_path):
    print(f'\n== {name}', flush=True)
    t0 = time.time()
    src = gs_io.read_source(path)
    print(f'  read {len(src)} splats in {time.time() - t0:.1f} s, kind {src.kind}, sh_degree {src.sh_degree}')
    summary = gs_io.describe(src)
    print('  describe:', json.dumps({k: v for k, v in summary.items() if k != 'ply_properties'}))
    check(summary['nonfinite'] == 0, f'{name}: no NaN/Inf values')

    # Version 1 (float32): every column must come back bit-exact.
    gs1_dir = os.path.join(out_dir, f'{name}_gs_v1')
    gs_format.write_gs(src, gs1_dir, version=1)
    back1 = gs_format.read_gs(gs1_dir)
    compare(src, back1, f'{name} gs v1 round-trip', tol=0.0)

    # Version 2 (quantised, default): error budget against the exact values.
    gs_dir = os.path.join(out_dir, f'{name}_gs')
    gs_format.write_gs(src, gs_dir)
    back = gs_format.read_gs(gs_dir)
    rep = gs_quant.error_report(src, back)
    print('  gs v2 errors:', json.dumps({k: float(f'{v:.4g}') for k, v in rep.items()}))
    for k, lim in test_gs_quant.BUDGET.items():
        check(rep[k] <= lim, f'{name} gs v2: {k} {rep[k]:.3g} <= {lim:g}')
    # Stable IDs: the row order on disk differs from the source order (Morton), and id.bin restores it.
    ids = np.fromfile(os.path.join(gs_dir, 'id.bin'), dtype='<u4')
    check(np.array_equal(np.sort(ids), np.arange(len(src))), f'{name}: id.bin is a permutation of 0..N-1')
    check(len(json.load(open(os.path.join(gs_dir, 'meta.json')))['chunks']) >= 1, f'{name}: chunks written')

    # LOD: a prefix of a chunk is a uniform sample of it. Octant histogram of the first quarter vs the whole chunk.
    with open(os.path.join(gs_dir, 'meta.json')) as f:
        chunks = json.load(f)['chunks']
    file_order = gs_format.read_gs(gs_dir, source_order=False)
    worst = 0.0
    for ch in chunks:
        if ch['count'] < 4000:
            continue
        pts = file_order.xyz[ch['start']:ch['start'] + ch['count']].astype(np.float64)
        mid = (np.array(ch['min']) + np.array(ch['max'])) / 2
        bins = lambda p: (p[:, 0] > mid[0]).astype(int) * 4 + (p[:, 1] > mid[1]) * 2 + (p[:, 2] > mid[2])  # noqa: E731
        full = np.bincount(bins(pts), minlength=8) / len(pts)
        part = pts[:len(pts) // 4]
        prefix = np.bincount(bins(part), minlength=8) / len(part)
        worst = max(worst, float(np.abs(prefix - full).max()))
    check(worst < 0.02, f'{name}: LOD prefix is a uniform sample (octant deviation {worst:.4f} < 0.02)')

    ply_path = os.path.join(out_dir, f'{name}_roundtrip.ply')
    gs_format.write_ply(back, ply_path)
    reread = gs_io.read_ply(ply_path)
    # rot is normalised again on read: allow float32 rounding only.
    compare(back, reread, f'{name} PLY round-trip of v2', tol=0.0, loose=('rot',))

    if ref_path and os.path.exists(ref_path):
        ref = gs_io.read_ply(ref_path)
        print(f'  reference: {ref_path} ({len(ref)} splats, kind {ref.kind}, sh_degree {ref.sh_degree})')
        check(len(ref) == len(src), f'{name}: splat count equals the reference')
        check(ref.sh_degree == src.sh_degree, f'{name}: SH degree equals the reference')
        if len(ref) == len(src) and ref.sh_degree == src.sh_degree:
            compare(src, ref, f'{name} vs reference', tol=1e-5)
    elif ref_path:
        print(f'  (no reference file {ref_path})')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref-dir', default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = lambda n: os.path.join(args.ref_dir, n) if args.ref_dir else None  # noqa: E731
    run_source('3dgs', os.path.join(args.data, '3DGS.ply'), args.out, ref('ref_3dgs.ply'))
    run_source('2dgs', os.path.join(args.data, '2dgs_cardboard_point_cloud.ply'), args.out, ref('ref_2dgs.ply'))
    run_source('fruit_sog', os.path.join(args.data, 'The Fruit Bowl.sog'), args.out, ref('fruit_ref.ply'))

    print(f'\n{len(failures)} failure(s)')
    for f in failures:
        print('  -', f)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
