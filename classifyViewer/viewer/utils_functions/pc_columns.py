"""
Writes the attribute columns of the chunked point cloud (see docs/POINTCLOUD_FORMAT.md).

The geometry (pc/geom.bin) is built once by las2pc. Every attribute lives in its own file
pc/col/<name>.bin, in the same point order as geom.bin, so computing features, classifying or
restoring a backup only rewrites the affected columns.

The join key between a LAS and the geometry is the POINT_ID. Memory is O(block): the LAS is read once,
in blocks, extracting every requested column from each block, and the columns are written in sequence.
Three ways of doing the join, tried in this order (the log says which one was used):

  fast   the LAS is in canonical order (row r of the LAS is row r of geom.bin, as written by las2pc
         and preserved by every later tool): the values are written as they are;
  merge  the LAS is an ordered subset of the cloud (e.g. a segment extracted for the classification):
         sequential merge with point_order.bin, points without a value get the missing marker;
  join   any other order: external join through bucket files in the temporary folder
         (LAS -> (pid, values) buckets by pid range, point_order.bin -> (pid, row) buckets, per-range
         dense join, (row, values) buckets by row range, per-range sequential write).

Usage:
  pc_columns.py --pc-dir DIR [--drop-all] [--las FILE (--all | --only a,b) [--prune]] [--prediction CLASSIFIED.las]
                [--memory-budget MB] [--temp-dir DIR]

Exit codes: 0 ok, 1 error, 3 the classified LAS does not match the point cloud (prediction mode).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import pipeline_common as pcm

PREDICTION_MISSING = 255
POINT_ID_NAME = 'POINT_ID'
SENTINEL = np.uint32(0xFFFFFFFF)


def _new_version():
    # Monotonic enough: two writes of the same column never share a token
    return str(time.time_ns() // 1000)


class Mismatch(Exception):
    """The LAS does not describe the points of the point cloud (strict / prediction mode)."""


def _column_type(name):
    return 'uint8' if name == 'prediction' else 'float32'


class PointCloud:
    def __init__(self, pc_dir):
        self.dir = pc_dir
        self.meta_path = os.path.join(pc_dir, 'meta.json')
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        self.n = int(self.meta['points'])
        self.order_path = os.path.join(pc_dir, self.meta['order']['file'])
        if os.path.getsize(self.order_path) != 4 * self.n:
            raise ValueError('point_order.bin does not match meta.json')

    def order_slice(self, a, b):
        return np.fromfile(self.order_path, dtype='<u4', count=b - a, offset=4 * a)

    def order_blocks(self, rows):
        with open(self.order_path, 'rb') as f:
            a = 0
            while a < self.n:
                m = min(rows, self.n - a)
                yield a, np.fromfile(f, dtype='<u4', count=m)
                a += m

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


class ColumnSink:
    """One output column: sequential writer to '<name>.bin.tmp', streaming min/max, atomic commit."""

    def __init__(self, pc, name, col_type):
        self.pc, self.name, self.col_type = pc, name, col_type
        self.dtype = np.dtype(np.uint8 if col_type == 'uint8' else np.float32)
        self.missing = PREDICTION_MISSING if col_type == 'uint8' else np.nan
        os.makedirs(os.path.join(pc.dir, 'col'), exist_ok=True)
        self.rel = f'col/{name}.bin'
        self.final = os.path.join(pc.dir, self.rel)
        self.tmp = self.final + '.tmp'
        self.w = pcm.SequentialWriter(self.tmp)
        self.rows = 0
        self.cmin = self.cmax = None
        self.collide = False

    def write_dense(self, values):
        out = np.ascontiguousarray(values, dtype=self.dtype)
        if out.size:
            if self.col_type == 'uint8':
                real = out[out != PREDICTION_MISSING]
            else:
                with np.errstate(all='ignore'):
                    real = out[np.isfinite(out)]
            if real.size:
                lo, hi = real.min(), real.max()
                self.cmin = lo if self.cmin is None else min(self.cmin, lo)
                self.cmax = hi if self.cmax is None else max(self.cmax, hi)
        self.w.write(out.tobytes())
        self.rows += out.size

    def check(self, raw):
        """Class ids >= 255 would collide with the missing marker (checked on the values read from the LAS)."""
        if self.col_type == 'uint8' and len(raw) and int(np.max(raw)) >= PREDICTION_MISSING:
            self.collide = True

    def write_missing(self, k):
        step = 1 << 22
        while k > 0:
            m = min(k, step)
            self.write_dense(np.full(m, self.missing, dtype=self.dtype))
            k -= m

    def abort(self):
        try:
            self.w.close()
        except Exception:
            pass
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def commit(self):
        if self.rows != self.pc.n:
            self.abort()
            raise RuntimeError(f"column '{self.name}': wrote {self.rows} rows, expected {self.pc.n}")
        self.w.close()
        os.replace(self.tmp, self.final)
        isint = self.col_type == 'uint8'
        entry = {'file': self.rel, 'type': self.col_type,
                 'min': (int(self.cmin) if isint else float(self.cmin)) if self.cmin is not None else None,
                 'max': (int(self.cmax) if isint else float(self.cmax)) if self.cmax is not None else None,
                 'version': _new_version()}
        if isint:
            entry['missing'] = PREDICTION_MISSING
        self.pc.meta.setdefault('columns', {})[self.name] = entry


class BucketWriter:
    """Records of one dtype appended to bucket files (open/append/close), flushed over a memory cap."""

    def __init__(self, folder, prefix, n_buckets, rec_dtype, cap_bytes):
        self.folder, self.prefix, self.nb, self.dt = folder, prefix, n_buckets, rec_dtype
        self.cap = cap_bytes
        self.buf = [[] for _ in range(n_buckets)]
        self.held = 0
        self.counts = np.zeros(n_buckets, dtype=np.int64)

    def path(self, k):
        return os.path.join(self.folder, f'{self.prefix}{k}.bin')

    def add(self, bucket, recs):
        if not len(recs):
            return
        order = np.argsort(bucket, kind='stable')
        b = bucket[order]
        r = recs[order]
        edges = np.flatnonzero(np.diff(b)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(b)]))
        for s, e in zip(starts, ends):
            k = int(b[s])
            chunk = r[s:e].tobytes()
            self.buf[k].append(chunk)
            self.held += len(chunk)
            self.counts[k] += e - s
        if self.held >= self.cap:
            self.flush()

    def flush(self):
        for k in range(self.nb):
            if self.buf[k]:
                with open(self.path(k), 'ab') as f:
                    f.write(b''.join(self.buf[k]))
                self.buf[k] = []
        self.held = 0

    def read(self, k, rows_per_block):
        p = self.path(k)
        if not os.path.exists(p):
            return
        with open(p, 'rb') as f:
            while True:
                a = np.fromfile(f, dtype=self.dt, count=rows_per_block)
                if not len(a):
                    break
                yield a

    def remove(self, k):
        try:
            os.remove(self.path(k))
        except OSError:
            pass


# ---------------------------------------------------------------------------- the three join paths

def _pids(blk):
    return np.ascontiguousarray(blk[POINT_ID_NAME]).astype('<u4', copy=False)


def fast_path(pc, las, names, sinks, rpb):
    """Row r of the LAS is row r of geom.bin. Returns False (sinks to be discarded) at the first difference."""
    if las.n != pc.n:
        return False
    dt_names = [POINT_ID_NAME] + names
    prog = pcm.Progress('columns', las.n)
    with open(pc.order_path, 'rb') as of:
        for start, blk in pcm.prefetched(las.blocks(dt_names, rpb)):
            order = np.fromfile(of, dtype='<u4', count=len(blk))
            if not np.array_equal(_pids(blk), order):
                return False
            for name, sink in zip(names, sinks):
                sink.check(blk[name])
                sink.write_dense(blk[name])
            prog.update(start + len(blk))
    return True


def merge_path(pc, las, names, sinks, rpb, budget):
    """The LAS is an ordered subset of the cloud. Returns the number of matched points or None."""
    if las.n > pc.n or las.n == 0:
        return None
    ratio = pc.n / las.n
    wmax = int(max(1 << 16, min(budget // 24, 64 << 20)))
    block = int(max(4096, min(rpb, wmax / (2.0 * ratio))))
    p = 0          # next unconsumed row of the cloud
    w = 0          # rows already written to the sinks
    matched = 0
    for start, blk in las.blocks([POINT_ID_NAME] + names, block):
        pid = _pids(blk)
        W = int(min(max(2 * ratio * len(pid), 1 << 16), wmax, pc.n - p))
        while True:
            if W <= 0:
                return None
            seg = pc.order_slice(p, p + W)
            srt = np.argsort(seg, kind='stable')
            ss = seg[srt]
            pos = np.searchsorted(ss, pid)
            pos[pos >= len(ss)] = len(ss) - 1
            if (ss[pos] == pid).all():
                break
            if p + W >= pc.n or W >= wmax:
                return None
            W = int(min(W * 2, wmax, pc.n - p))
        rows = p + srt[pos].astype(np.int64)
        if len(rows) > 1 and (np.diff(rows) <= 0).any():
            return None
        span = int(rows[-1]) + 1 - w
        idx = rows - w
        for name, sink in zip(names, sinks):
            sink.check(blk[name])
            out = np.full(span, sink.missing, dtype=sink.dtype)
            out[idx] = blk[name]
            sink.write_dense(out)
        matched += len(rows)
        w = p = int(rows[-1]) + 1
    for sink in sinks:
        sink.write_missing(pc.n - w)
    return matched


def join_path(pc, las, names, sinks, budget, temp_root, strict, rpb):
    """External join through bucket files. Returns the number of matched points."""
    nc = len(names)
    # pid range of the cloud
    pmin, pmax = 0xFFFFFFFF, 0
    for _, o in pc.order_blocks(1 << 22):
        pmin, pmax = min(pmin, int(o.min())), max(pmax, int(o.max()))
    span = pmax - pmin + 1
    W = int(max(1 << 16, min(span, budget // 4 // 5)))           # dense row (4 B) + seen flag (1 B) per pid
    K = -(-span // W)
    col_item = [np.dtype(np.uint8 if s.col_type == 'uint8' else np.float32).itemsize for s in sinks]
    Rw = int(max(1 << 16, min(pc.n, budget // 4 // (sum(col_item) + 1))))
    Rb = -(-pc.n // Rw)
    need = pc.n * 8 + las.n * (4 + 4 * nc) * 2
    pcm.log_phase(f'[pc_columns] join: {K} pid range(s) of {W} ids, {Rb} row range(s) of {Rw} rows, '
                  f'~{need / 1e9:.1f} GB of temporary files')

    with pcm.TempWorkDir(temp_root, prefix='pccol_', need_bytes=need) as tmp:
        cap = max(8 << 20, budget // 8)
        o_dt = np.dtype([('pid', '<u4'), ('row', '<u4')])
        l_dt = np.dtype([('pid', '<u4'), ('v', '<f4', (nc,))])
        r_dt = np.dtype([('row', '<u4'), ('v', '<f4', (nc,))])
        ob = BucketWriter(tmp.path, 'o', K, o_dt, cap)
        lb = BucketWriter(tmp.path, 'l', K, l_dt, cap)
        rb = BucketWriter(tmp.path, 'r', Rb, r_dt, cap)

        # (b) point_order.bin -> (pid, row) buckets
        for a, o in pc.order_blocks(1 << 21):
            recs = np.empty(len(o), dtype=o_dt)
            recs['pid'] = o
            recs['row'] = np.arange(a, a + len(o), dtype=np.uint32)
            ob.add(((o.astype(np.int64) - pmin) // W), recs)
        ob.flush()

        # (a) LAS -> (pid, values) buckets
        bad = 0
        for start, blk in pcm.prefetched(las.blocks([POINT_ID_NAME] + names, min(rpb, 1 << 21))):
            pid = _pids(blk)
            inside = (pid >= pmin) & (pid <= pmax)
            bad += int((~inside).sum())
            recs = np.empty(int(inside.sum()), dtype=l_dt)
            recs['pid'] = pid[inside]
            for j, name in enumerate(names):
                sinks[j].check(blk[name])
                recs['v'][:, j] = blk[name][inside]
            lb.add((pid[inside].astype(np.int64) - pmin) // W, recs)
        lb.flush()
        if strict and bad:
            raise Mismatch(f'{bad} POINT_ID of the classified LAS are not in the point cloud')

        # (c) per pid range: dense join, emit (row, values) by row range
        matched = 0
        for k in range(K):
            lo = pmin + k * W
            dense = np.full(W, SENTINEL, dtype=np.uint32)
            for o in ob.read(k, 1 << 21):
                dense[o['pid'] - lo] = o['row']
            ob.remove(k)
            seen = np.zeros(W, dtype=np.uint8) if strict else None
            for l in lb.read(k, 1 << 21):
                idx = l['pid'] - lo
                rows = dense[idx]
                ok = rows != SENTINEL
                if not ok.all():
                    bad += int((~ok).sum())
                    if strict:
                        raise Mismatch(f'{bad} POINT_ID of the classified LAS are not in the point cloud')
                if strict:
                    u = np.unique(idx)
                    if len(u) != len(idx) or seen[u].any():
                        raise Mismatch('duplicated POINT_ID in the classified LAS')
                    seen[u] = 1
                recs = np.empty(int(ok.sum()), dtype=r_dt)
                recs['row'] = rows[ok]
                recs['v'] = l['v'][ok]
                matched += len(recs)
                rb.add(rows[ok].astype(np.int64) // Rw, recs)
            lb.remove(k)
            del dense
        rb.flush()

        # (d) per row range: fill the slice of every column and write it in sequence
        for r in range(Rb):
            lo, hi = r * Rw, min(pc.n, (r + 1) * Rw)
            cols = [np.full(hi - lo, s.missing, dtype=s.dtype) for s in sinks]
            for rec in rb.read(r, 1 << 21):
                idx = rec['row'] - lo
                for j in range(nc):
                    cols[j][idx] = rec['v'][:, j].astype(cols[j].dtype, copy=False)
            rb.remove(r)
            for s, c in zip(sinks, cols):
                s.write_dense(c)
    if bad and not strict:
        pcm.log_phase(f'[pc_columns] {bad} POINT_ID of the LAS are not in the point cloud (ignored)')
    return matched


def scatter_columns(pc, las, cols, budget, temp_root, strict=False, force_path=None):
    """
    Writes the columns `cols` ([(name, type)]) from the LAS. Returns {'path': ..., 'matched': ...}.
    Raises Mismatch in strict mode when the LAS does not belong to the cloud.
    """
    names = [c[0] for c in cols]
    rpb = las.block_rows_for(budget * 0.3 / 3, per_row_extra=8 + 4 * len(names))   # raw block, x3 with prefetch
    sinks = [ColumnSink(pc, n, t) for n, t in cols]
    pending = [s.tmp for s in sinks]
    pcm.install_term_cleanup(lambda: pending)

    def reset():
        for s in sinks:
            s.abort()
        return [ColumnSink(pc, n, t) for n, t in cols]

    try:
        path = None
        matched = None
        if force_path in (None, 'fast') and las.n == pc.n:
            if fast_path(pc, las, names, sinks, rpb):
                path, matched = 'fast', pc.n
            else:
                sinks = reset()
                if force_path == 'fast':
                    raise RuntimeError('forced fast path but the LAS is not in canonical order')
        if path is None and force_path in (None, 'merge'):
            m = merge_path(pc, las, names, sinks, rpb, budget)
            if m is not None:
                path, matched = 'merge', m
            else:
                sinks = reset()
        if path is None:
            matched = join_path(pc, las, names, sinks, budget, temp_root, strict, rpb)
            path = 'join'
        for s in sinks:
            s.commit()
        return {'path': path, 'matched': int(matched), 'collide': any(s.collide for s in sinks)}
    except BaseException:
        for s in sinks:
            s.abort()
        raise


# ---------------------------------------------------------------------------- operations

def update_from_las(pc, las_path, names, budget, temp_root, prune=False, force_path=None):
    las = pcm.LasFile(las_path)
    if POINT_ID_NAME not in las.extra_names:
        raise ValueError(f'{POINT_ID_NAME} extra byte not found in {las_path}')
    fields = las.extra_names
    if names is None:
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
    if names:
        info = scatter_columns(pc, las, [(n, _column_type(n)) for n in names], budget, temp_root, force_path=force_path)
        print(f"[pc_columns] path: {info['path']}", flush=True)
        for name in names:
            print(f"  column {name}: {info['matched']} values", flush=True)
        if info['matched'] < pc.n:
            print(f"[Info] {pc.n - info['matched']} points of the cloud have no value", flush=True)
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


def update_prediction(pc, las_path, budget, temp_root, force_path=None):
    las = pcm.LasFile(las_path)
    if POINT_ID_NAME not in las.extra_names or 'prediction' not in las.extra_names:
        print('PC_COLUMNS_MISMATCH: classified LAS has no POINT_ID/prediction', flush=True)
        return 3
    try:
        info = scatter_columns(pc, las, [('prediction', 'uint8')], budget, temp_root, strict=True, force_path=force_path)
    except Mismatch as e:
        print(f'PC_COLUMNS_MISMATCH: {e}', flush=True)
        return 3
    print(f"[pc_columns] path: {info['path']}", flush=True)
    if info['collide']:
        print('[Warning] prediction values >= 255 collide with the "no prediction" marker', flush=True)
    print(f"  column prediction: {info['matched']} of {pc.n} points", flush=True)
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
    ap.add_argument('--force-path', choices=['fast', 'merge', 'join'], help=argparse.SUPPRESS)
    pcm.add_common_args(ap)
    args = ap.parse_args()

    budget = pcm.memory_budget_bytes(args.memory_budget)
    pc = PointCloud(args.pc_dir)
    code = 0
    if args.drop_all:
        pc.drop_all()
    if args.all or args.only:
        if not args.las:
            ap.error('--las is required with --all/--only')
        names = None if args.all else [s for s in args.only.split(',') if s]
        update_from_las(pc, args.las, names, budget, args.temp_dir, prune=args.prune, force_path=args.force_path)
    if args.prediction:
        code = update_prediction(pc, args.prediction, budget, args.temp_dir, force_path=args.force_path)
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
