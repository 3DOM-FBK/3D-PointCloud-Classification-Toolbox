#!/usr/bin/env python3
"""
End-to-end checks of the column pipeline.

  test_columns.py subset <features.las> <las2pc> <pc_columns.py> <workdir>
      las2pc -> columns --all -> prediction from a shuffled SUBSET LAS (+ mismatch detection)
  test_columns.py big <las2pc> <pc_columns.py> <RF_classify.py> <workdir> [n_points]
      synthetic cloud with more than 2^24 points: POINT_ID must survive RF_classify.py and the
      prediction column must be scattered to the right points
"""
import json
import os
import pickle
import subprocess
import sys

import laspy
import numpy as np

ok = True


def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg, flush=True)
    ok &= bool(cond)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def column(pc_dir, meta, name):
    dt = '<f4' if meta['columns'][name]['type'] == 'float32' else 'u1'
    return np.fromfile(f"{pc_dir}/{meta['columns'][name]['file']}", dtype=dt)


def make_las(path, xyz, extras):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001] * 3
    header.offsets = xyz.min(axis=0)
    header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=a.dtype) for n, a in extras.items()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red = las.green = las.blue = np.full(len(xyz), 40000, dtype=np.uint16)
    for n, a in extras.items():
        las[n] = a
    las.write(path)


def test_subset(src, las2pc, pcc, work):
    pc = f"{work}/pc"
    rc, out = run([las2pc, '-i', src, '-o', pc])
    check(rc == 0, "las2pc")
    rc, out = run([sys.executable, pcc, '--pc-dir', pc, '--las', src, '--all'])
    check(rc == 0, "pc_columns --all")
    meta = json.load(open(f"{pc}/meta.json"))
    check(set(meta['columns']) >= {'NormalX', 'NormalY', 'NormalZ'}, "columns registered in meta.json")

    las = laspy.read(src)
    n = len(las.points)
    rng = np.random.default_rng(1)
    sel = rng.permutation(n)[: n // 5]
    pid = np.asarray(las['POINT_ID'])[sel]
    pred = (pid % 5).astype(np.uint8)
    sub = f"{work}/classified.las"
    make_las(sub, np.stack([las.x, las.y, las.z], 1)[sel], {'POINT_ID': pid.astype(np.uint32), 'prediction': pred})
    rc, out = run([sys.executable, pcc, '--pc-dir', pc, '--prediction', sub])
    check(rc == 0, "prediction from a shuffled subset")
    meta = json.load(open(f"{pc}/meta.json"))
    col = column(pc, meta, 'prediction')
    order = np.fromfile(f"{pc}/point_order.bin", dtype='<u4')
    expect = np.full(n, 255, dtype=np.uint8)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    expect[inv[pid]] = pred
    check(np.array_equal(col, expect), "prediction column == expected by POINT_ID (255 elsewhere)")
    check(meta['columns']['prediction']['missing'] == 255, "missing marker documented")

    bad = f"{work}/bad.las"
    make_las(bad, np.stack([las.x, las.y, las.z], 1)[:100], {'POINT_ID': (np.arange(100) + 10 * n).astype(np.uint32),
                                                              'prediction': np.zeros(100, np.uint8)})
    rc, out = run([sys.executable, pcc, '--pc-dir', pc, '--prediction', bad])
    check(rc == 3 and 'PC_COLUMNS_MISMATCH' in out, "mismatching LAS is rejected (exit 3)")

    rc, out = run([sys.executable, pcc, '--pc-dir', pc, '--drop-all'])
    meta = json.load(open(f"{pc}/meta.json"))
    check(rc == 0 and meta['columns'] == {} and not os.listdir(f"{pc}/col"), "--drop-all")


def test_big(las2pc, pcc, rf, work, n):
    n = int(n)
    rng = np.random.default_rng(7)
    xyz = rng.random((n, 3)) * [100, 100, 20]
    f1 = (xyz[:, 0] / 100).astype(np.float32)
    f2 = (xyz[:, 1] / 100).astype(np.float32)
    # POINT_ID: a permutation, so the file order is NOT the id order
    pid = rng.permutation(n).astype(np.uint32)
    src = f"{work}/big.las"
    make_las(src, xyz, {'POINT_ID': pid, 'f1': f1, 'f2': f2})
    check(int(pid.max()) > 16777216, f"max POINT_ID {int(pid.max())} > 2^24")

    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=5, max_depth=6, random_state=0)
    idx = rng.integers(0, n, 20000)
    labels = ((f1[idx] > 0.5).astype(int) + 2 * (f2[idx] > 0.5)).astype(int)
    m.fit(np.stack([f1[idx], f2[idx]], 1), labels)
    pickle.dump(m, open(f"{work}/model.pkl", 'wb'))

    out_las = f"{work}/big_classified.las"
    rc, out = run([sys.executable, rf, '--selected_features', 'f1', 'f2', '--model', f"{work}/model.pkl",
                   '--test_filepath', src, '--output_classify_name', out_las])
    check(rc == 0, "RF_classify.py")
    if rc != 0:
        print(out[-2000:])
    res = laspy.read(out_las)
    rpid = np.asarray(res['POINT_ID'])
    check(np.array_equal(rpid, pid), "POINT_ID preserved (uint32) by RF_classify.py")

    pc = f"{work}/pc_big"
    rc, _ = run([las2pc, '-i', src, '-o', pc])
    check(rc == 0, "las2pc on > 2^24 points")
    rc, out = run([sys.executable, pcc, '--pc-dir', pc, '--prediction', out_las])
    check(rc == 0, "prediction column on > 2^24 points")
    meta = json.load(open(f"{pc}/meta.json"))
    col = column(pc, meta, 'prediction')
    order = np.fromfile(f"{pc}/point_order.bin", dtype='<u4')
    expect_by_pid = np.empty(n, dtype=np.uint8)
    expect_by_pid[rpid] = np.asarray(res['prediction'])
    check(np.array_equal(col, expect_by_pid[order]), "prediction scattered to the right points")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'subset':
        test_subset(*sys.argv[2:6])
    else:
        test_big(*sys.argv[2:7])
    print("RESULT:", "ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
