"""Compares two renders pixel by pixel (the HUD box of the browser page is excluded).

Writes a difference image (per-pixel max channel difference, scaled x4) next to the output name.
Usage: python compare_renders.py BROWSER.png REFERENCE.png OUT_DIFF.png
"""
import sys

import numpy as np
from PIL import Image

HUD = (0, 0, 530, 85)    # x0, y0, x1, y1 of the HUD box in the browser page (covers the longest line)


def main():
    a = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int)
    b = np.asarray(Image.open(sys.argv[2]).convert('RGB')).astype(int)
    if a.shape != b.shape:
        raise SystemExit(f'shape mismatch {a.shape} vs {b.shape}')
    d = np.abs(a - b).max(axis=2)
    x0, y0, x1, y1 = HUD
    mask = np.ones(d.shape, bool)
    mask[y0:y1, x0:x1] = False
    dm = d[mask]
    covered = (np.abs(b - np.array([23, 24, 28])).max(axis=2) > 2)[mask]   # pixels touched by splats in the reference
    print(f'pixels compared: {dm.size}')
    print(f'max diff {dm.max()}  mean diff {dm.mean():.3f}')
    print(f'pixels with diff > 2: {(dm > 2).mean() * 100:.2f} %   > 16: {(dm > 16).mean() * 100:.3f} %')
    print(f'splat-covered pixels (reference): {covered.mean() * 100:.1f} %, '
          f'their mean diff {dm[covered].mean():.3f}, max {dm[covered].max()}')
    diff = np.clip(d * 4, 0, 255).astype(np.uint8)
    diff[y0:y1, x0:x1] = 0
    Image.fromarray(diff).save(sys.argv[3])


if __name__ == '__main__':
    main()
