"""Convert a Gaussian splat file (3DGS PLY, 2DGS PLY, SOG v2) into the internal `gs` folder.

Usage:
  python gs_convert.py INPUT OUTPUT_DIR            # write the gs folder
  python gs_convert.py INPUT OUTPUT_DIR --ply      # also write <OUTPUT_DIR>/roundtrip.ply (verification)

See docs/GAUSSIAN_SPLAT_FORMAT.md for the format.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gs_format  # noqa: E402
import gs_io  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output_dir')
    ap.add_argument('--ply', action='store_true', help='also write a 3DGS/2DGS PLY from the gs folder')
    ap.add_argument('--exact', action='store_true', help='write format version 1 (float32, no quantisation)')
    args = ap.parse_args()

    t0 = time.time()
    src = gs_io.read_source(args.input)
    summary = gs_io.describe(src)
    print(f'read {summary["count"]} splats ({src.kind}, SH degree {src.sh_degree}) in {time.time() - t0:.1f} s')
    if summary['nonfinite']:
        raise SystemExit(f'{summary["nonfinite"]} non-finite values in the input, refusing to convert')

    t1 = time.time()
    meta = gs_format.write_gs(src, args.output_dir, version=1 if args.exact else gs_format.FORMAT_VERSION)
    print(f'wrote gs folder {args.output_dir} ({len(meta["chunks"])} chunks) in {time.time() - t1:.1f} s')

    if args.ply:
        ply = os.path.join(args.output_dir, 'roundtrip.ply')
        gs_format.write_ply(gs_format.read_gs(args.output_dir), ply)
        print(f'wrote {ply}')


if __name__ == '__main__':
    main()
