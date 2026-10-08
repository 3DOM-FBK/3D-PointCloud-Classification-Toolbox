"""
Writes the attribute columns of the chunked point cloud (see docs/POINTCLOUD_FORMAT.md).

The geometry (pc/geom.bin) is built once by las2pc. Every attribute lives in its own file
pc/col/<name>.bin, in the same point order as geom.bin, so computing features, classifying or
restoring a backup only rewrites the affected columns.

The join key between a LAS and the geometry is always POINT_ID (the GPU feature extractor rewrites
the records tile by tile, so the position in the file is meaningless).

Usage:
  pc_columns.py --pc-dir DIR [--drop-all] [--las FILE (--all | --only a,b) [--prune]] [--prediction CLASSIFIED.las]

Exit codes: 0 ok, 1 error, 3 the classified LAS does not match the point cloud (prediction mode).
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

PREDICTION_MISSING = 255
POINT_ID_NAME = 'POINT_ID'

# LAS Extra Bytes data_type -> numpy dtype (scalars only)
EXTRA_DTYPES = {
    1: '<u1', 2: '<i1', 3: '<u2', 4: '<i2', 5: '<u4', 6: '<i4',
    7: '<u8', 8: '<i8', 9: '<f4', 10: '<f8',
}
EXTRA_SIZES = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 8, 8: 8, 9: 4, 10: 8, 11: 2, 12: 2, 13: 4, 14: 4,
    15: 8, 16: 8, 17: 16, 18: 16, 19: 8, 20: 16, 21: 3, 22: 3, 23: 6, 24: 6, 25: 12, 26: 12,
    27: 24, 28: 24, 29: 12, 30: 24,
}
BASE_RECORD_SIZES = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}


def _new_version():
    # Monotonic enough: two writes of the same column never share a token
    return str(time.time_ns() // 1000)


def read_las_extra_fields(path):
    """
    Opens a LAS with np.memmap and returns (records, fields) where `records` is a structured memmap
    exposing only the Extra Bytes scalar fields (zero-copy views) and `fields` maps name -> numpy dtype.
    """
    with open(path, 'rb') as f:
        head = f.read(375)
    if head[:4] != b'LASF':
        raise ValueError(f'Not a LAS file: {path}')
    header_size, = struct.unpack_from('<H', head, 94)
    data_offset, = struct.unpack_from('<I', head, 96)
    num_vlrs, = struct.unpack_from('<I', head, 100)
    fmt = head[104]
    if fmt & 0xC0:
        raise ValueError('Compressed LAS (LAZ) is not supported')
    fmt &= 0x3F
    record_len, = struct.unpack_from('<H', head, 105)
    n_points, = struct.unpack_from('<I', head, 107)
    if n_points == 0 and head[25] >= 4:
        n_points, = struct.unpack_from('<Q', head, 247)
    base = BASE_RECORD_SIZES.get(fmt)
    if base is None:
        raise ValueError(f'Unsupported LAS point format {fmt}')

    names, formats, offsets = [], [], []
    with open(path, 'rb') as f:
        f.seek(header_size)
        for _ in range(num_vlrs):
            hdr = f.read(54)
            if len(hdr) < 54:
                break
            user_id = hdr[2:18].rstrip(b'\x00').decode('latin-1')
            record_id, = struct.unpack_from('<H', hdr, 18)
            length, = struct.unpack_from('<H', hdr, 20)
            body = f.read(length)
            if user_id.startswith('LASF_Spec') and record_id == 4:
                cur = base
                for i in range(length // 192):
                    rec = body[i * 192:(i + 1) * 192]
                    dtype, options = rec[2], rec[3]
                    name = rec[4:36].split(b'\x00')[0].decode('latin-1').strip()
                    size = options if dtype == 0 else EXTRA_SIZES.get(dtype, 0)
                    if dtype in EXTRA_DTYPES and name and name not in names:
                        names.append(name)
                        formats.append(EXTRA_DTYPES[dtype])
                        offsets.append(cur)
                    cur += size
                break
    if not names:
        return None, {}, n_points
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': record_len})
    records = np.memmap(path, dtype=dt, mode='r', offset=data_offset, shape=(n_points,))
    return records, {n: np.dtype(f) for n, f in zip(names, formats)}, n_points


class PointCloud:
    def __init__(self, pc_dir):
        self.dir = pc_dir
        self.meta_path = os.path.join(pc_dir, 'meta.json')
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        self.n = int(self.meta['points'])
        self.order = np.fromfile(os.path.join(pc_dir, self.meta['order']['file']), dtype='<u4')
        if len(self.order) != self.n:
            raise ValueError('point_order.bin does not match meta.json')
        self._inverse = None

    def rows_for(self, point_ids):
        """Position (in geom order) of every POINT_ID, -1 when the point is not in the cloud."""
        pid = np.asarray(point_ids, dtype=np.uint32)
        max_pid = int(self.order.max()) if self.n else 0
        if max_pid < 4 * self.n + 1024:
            if self._inverse is None:
                inv = np.full(max_pid + 1, -1, dtype=np.int64)
                inv[self.order] = np.arange(self.n, dtype=np.int64)
                self._inverse = inv
            rows = np.full(len(pid), -1, dtype=np.int64)
            ok = pid <= max_pid
            rows[ok] = self._inverse[pid[ok]]
            return rows
        sorter = np.argsort(self.order, kind='stable')
        sorted_ids = self.order[sorter]
        pos = np.searchsorted(sorted_ids, pid)
        pos[pos >= self.n] = self.n - 1
        return np.where(sorted_ids[pos] == pid, sorter[pos], -1)

    def write_column(self, name, values, rows, col_type):
        """Scatter `values` (aligned with `rows`) into a fresh column and register it in meta.json."""
        missing = PREDICTION_MISSING if col_type == 'uint8' else np.nan
        out = np.full(self.n, missing, dtype=np.uint8 if col_type == 'uint8' else np.float32)
        valid = rows >= 0
        n_bad = int(len(rows) - valid.sum())
        if n_bad:
            print(f"[Warning] column '{name}': {n_bad} POINT_ID of the LAS are not in the point cloud (ignored)", flush=True)
        out[rows[valid]] = values[valid].astype(out.dtype, copy=False)
        n_set = int(valid.sum())
        if n_set < self.n:
            print(f"[Info] column '{name}': {self.n - n_set} points of the cloud have no value", flush=True)

        col_dir = os.path.join(self.dir, 'col')
        os.makedirs(col_dir, exist_ok=True)
        rel = f'col/{name}.bin'
        final = os.path.join(self.dir, rel)
        tmp = final + '.tmp'
        out.tofile(tmp)
        os.replace(tmp, final)

        if col_type == 'uint8':
            real = out[out != PREDICTION_MISSING]
            cmin = int(real.min()) if real.size else None
            cmax = int(real.max()) if real.size else None
        else:
            with np.errstate(all='ignore'):
                finite = out[np.isfinite(out)]
            cmin = float(finite.min()) if finite.size else None
            cmax = float(finite.max()) if finite.size else None
        entry = {'file': rel, 'type': col_type, 'min': cmin, 'max': cmax, 'version': _new_version()}
        if col_type == 'uint8':
            entry['missing'] = PREDICTION_MISSING
        self.meta.setdefault('columns', {})[name] = entry

    def drop_all(self):
        col_dir = os.path.join(self.dir, 'col')
        if os.path.isdir(col_dir):
            for fn in os.listdir(col_dir):
                try:
                    os.remove(os.path.join(col_dir, fn))
                except OSError:
                    pass
        self.meta['columns'] = {}

    def save_meta(self):
        tmp = self.meta_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.meta, f, separators=(',', ':'))
        os.replace(tmp, self.meta_path)


def _column_type(name):
    return 'uint8' if name == 'prediction' else 'float32'


def update_from_las(pc, las_path, names, prune=False):
    records, fields, n_las = read_las_extra_fields(las_path)
    if records is None or POINT_ID_NAME not in fields:
        raise ValueError(f'{POINT_ID_NAME} extra byte not found in {las_path}')
    pids = np.asarray(records[POINT_ID_NAME])
    rows = pc.rows_for(pids)
    if (names is None):
        names = [n for n in fields if n != POINT_ID_NAME]
    else:
        missing = [n for n in names if n not in fields]
        if missing:
            print(f"[Warning] attributes not found in the LAS (skipped): {missing}", flush=True)
        names = [n for n in names if n in fields]
        # Extra Bytes the caller did not name but the cloud does not know yet are added anyway
        known = pc.meta.get('columns', {})
        names += [n for n in fields if n != POINT_ID_NAME and n not in known and n not in names]
    t0 = time.time()
    for name in names:
        values = np.asarray(records[name])
        pc.write_column(name, values, rows, _column_type(name))
        print(f"  column {name}: {int((rows >= 0).sum())} values", flush=True)
    if prune:
        # Columns that no longer exist in the LAS (the extractor rewrites the Extra Bytes) are dropped;
        # 'prediction' comes from the classification, not from features.las: it stays.
        for stale in [n for n in pc.meta.get('columns', {}) if n not in fields and n != 'prediction']:
            try:
                os.remove(os.path.join(pc.dir, pc.meta['columns'][stale]['file']))
            except OSError:
                pass
            del pc.meta['columns'][stale]
            print(f"  column {stale} dropped (not in the LAS any more)", flush=True)
    print(f"[pc_columns] {len(names)} column(s) written in {time.time() - t0:.2f} s", flush=True)


def update_prediction(pc, las_path):
    records, fields, n_las = read_las_extra_fields(las_path)
    if records is None or POINT_ID_NAME not in fields or 'prediction' not in fields:
        print('PC_COLUMNS_MISMATCH: classified LAS has no POINT_ID/prediction', flush=True)
        return 3
    pids = np.asarray(records[POINT_ID_NAME])
    rows = pc.rows_for(pids)
    if (rows < 0).any():
        print(f'PC_COLUMNS_MISMATCH: {int((rows < 0).sum())} POINT_ID of the classified LAS are not in the point cloud', flush=True)
        return 3
    if len(np.unique(pids)) != len(pids):
        print('PC_COLUMNS_MISMATCH: duplicated POINT_ID in the classified LAS', flush=True)
        return 3
    pred = np.asarray(records['prediction'])
    if pred.size and int(pred.max()) >= PREDICTION_MISSING:
        print('[Warning] prediction values >= 255 collide with the "no prediction" marker', flush=True)
    pc.write_column('prediction', pred, rows, 'uint8')
    print(f"  column prediction: {len(rows)} of {pc.n} points", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc-dir', required=True)
    ap.add_argument('--las')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--prediction')
    ap.add_argument('--drop-all', action='store_true')
    ap.add_argument('--prune', action='store_true', help='drop columns that are not in the LAS any more')
    args = ap.parse_args()

    pc = PointCloud(args.pc_dir)
    code = 0
    if args.drop_all:
        pc.drop_all()
    if args.all or args.only:
        if not args.las:
            ap.error('--las is required with --all/--only')
        names = None if args.all else [s for s in args.only.split(',') if s]
        update_from_las(pc, args.las, names, prune=args.prune)
    if args.prediction:
        code = update_prediction(pc, args.prediction)
    if code == 0:
        pc.save_meta()
        print('pc_columns completed', flush=True)
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:  # surface the reason as the last stdout line (JobManager reports it)
        print(f'ERROR: {e}', flush=True)
        sys.exit(1)
