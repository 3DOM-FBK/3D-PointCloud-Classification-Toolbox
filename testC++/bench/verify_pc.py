#!/usr/bin/env python3
"""
Verifies a point cloud folder produced by las2pc against its source LAS.

Checks:
  1. every POINT_ID appears exactly once in geom.bin (and point_order.bin == geom POINT_ID);
  2. decoded coordinates ((q + qMin) * scale + offset) equal the LAS coordinates of that POINT_ID;
  3. the level prefix lengths match meta.json levelPoints (head block and body layout);
  4. stratification: inside a chunk, level l holds at most one point per 2^(base+l) cell, and every
     point of a later level falls on a cell already occupied by an earlier level;
  5. optional (--columns): every column in meta.json equals the LAS extra byte of the same name.

Usage: verify_pc.py <features.las> <pc_dir> [--columns]
"""
import json
import sys
import numpy as np
import laspy


def main():
    las_path, pc_dir = sys.argv[1], sys.argv[2].rstrip("/")
    check_cols = "--columns" in sys.argv
    meta = json.load(open(f"{pc_dir}/meta.json"))
    n = meta["points"]
    las = laspy.read(las_path)
    assert len(las.points) == n, f"point count {len(las.points)} != {n}"
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    geom = np.fromfile(f"{pc_dir}/geom.bin", dtype=np.dtype([("q", "<i4", 3), ("rgb", "u1", 3), ("pad", "u1"), ("pid", "<u4")]))
    order = np.fromfile(f"{pc_dir}/point_order.bin", dtype="<u4")
    check(len(geom) == n and len(order) == n, "geom.bin / point_order.bin sizes")
    check(np.array_equal(geom["pid"], order), "point_order.bin == geom POINT_ID")

    src_pid = np.asarray(las["POINT_ID"], dtype=np.uint32) if "POINT_ID" in las.point_format.dimension_names else np.arange(n, dtype=np.uint32)
    check(np.array_equal(np.sort(geom["pid"]), np.sort(src_pid)), "every POINT_ID exactly once")

    # coordinates
    pos = np.empty(n, dtype=np.int64)
    pos[np.argsort(src_pid, kind="stable")] = np.arange(n)  # src row for rank
    sorter = np.argsort(src_pid)
    rows = sorter[np.searchsorted(src_pid[sorter], geom["pid"])]
    scale, offset, qmin = np.array(meta["scale"]), np.array(meta["offset"]), np.array(meta["qMin"], dtype=np.int64)
    src_x = np.stack([las.X, las.Y, las.Z], axis=1).astype(np.int64)[rows]
    check(np.array_equal(geom["q"].astype(np.int64) + qmin, src_x), "decoded integer coordinates == LAS X/Y/Z")
    dec = (geom["q"].astype(np.float64) + qmin) * scale + offset
    ref = np.stack([las.x, las.y, las.z], axis=1)[rows]
    check(np.abs(dec - ref).max() < 1e-6, f"decoded real coordinates (max err {np.abs(dec - ref).max():.2e})")
    bb = meta["boundingBox"]
    check(np.all(dec.min(0) >= np.array(bb["min"]) - 1e-9) and np.all(dec.max(0) <= np.array(bb["max"]) + 1e-9), "points inside bbox")

    # layout & levels
    H, base, nlev = meta["levels"]["head"], meta["levels"]["base"], meta["levels"]["count"]
    chunks = meta["chunks"]
    check(sum(c["points"] for c in chunks) == n, "sum of chunk points == N")
    head_total = sum(sum(c["levelPoints"][:H]) for c in chunks)
    check(head_total == meta["head"]["points"], "head.points == sum of head levels")
    bad_strat = 0
    bad_layout = 0
    local = dec - np.array(bb["min"])
    for c in chunks:
        lp = c["levelPoints"]
        hp = sum(lp[:H])
        if sum(lp) != c["points"]:
            bad_layout += 1
        head_rng = (c["headOffset"], c["headOffset"] + hp)
        body_rng = (c["bodyOffset"], c["bodyOffset"] + c["points"] - hp)
        if head_rng[0] < 0 or head_rng[1] > meta["head"]["points"] or body_rng[0] < meta["head"]["points"] or body_rng[1] > n:
            bad_layout += 1
        idx = np.concatenate([np.arange(*head_rng), np.arange(*body_rng)])
        pts = local[idx] - np.array(c["cubeMin"])
        start = 0
        occupied = []
        for l in range(nlev):
            seg = pts[start:start + lp[l]]
            start += lp[l]
            if l < nlev - 1:
                R = 1 << (base + l)
                cells = np.clip((seg / c["size"] * R).astype(np.int64), 0, R - 1)
                key = (cells[:, 2] * R + cells[:, 1]) * R + cells[:, 0]
                if len(np.unique(key)) != len(key):
                    bad_strat += 1
                occupied.append((R, set(key.tolist())))
            # later levels must fall on occupied cells of every earlier level
            if l > 0:
                for (R, occ) in occupied:
                    if R == (1 << (base + l)) and l < nlev - 1:
                        continue
                    cells = np.clip((seg / c["size"] * R).astype(np.int64), 0, R - 1)
                    key = (cells[:, 2] * R + cells[:, 1]) * R + cells[:, 0]
                    if not set(key.tolist()).issubset(occ):
                        bad_strat += 1
                        break
    check(bad_layout == 0, "chunk offsets / level sums consistent")
    check(bad_strat == 0, f"stratification property in {len(chunks)} chunks")

    if check_cols:
        for name, col in meta.get("columns", {}).items():
            dt = {"float32": "<f4", "uint8": "u1"}[col["type"]]
            arr = np.fromfile(f"{pc_dir}/{col['file']}", dtype=dt)
            check(len(arr) == n, f"column {name}: length")
            if name in las.point_format.dimension_names:
                ref = np.asarray(las[name])[rows]
                same = np.array_equal(arr, ref.astype(arr.dtype), equal_nan=arr.dtype.kind == "f")
                check(same, f"column {name}: values == LAS")

    print("RESULT:", "ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
