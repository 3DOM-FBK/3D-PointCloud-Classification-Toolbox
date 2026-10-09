"""
Shared helpers of the out-of-core Python tools (pc_columns.py, RF_classify.py, ...).

  * memory budget          (--memory-budget MB, default 50% of min(cgroup limit, MemAvailable))
  * temporary work folder  (--temp-dir, one sub-folder per job, removed on exit / error / SIGTERM)
  * LAS reader by blocks   (header, VLRs, Extra Bytes, sequential pread with DONTNEED behind the cursor)

Nothing here loads a whole file in memory.
"""
import atexit
import os
import shutil
import signal
import struct
import sys
import tempfile
import time

import numpy as np

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
RGB_OFFSETS = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}

DEFAULT_TEMP_DIR = '/tmp/pipeline_work'


# ---------------------------------------------------------------------------- memory budget

def available_memory_bytes():
    """min(cgroup limit, MemAvailable); None when it cannot be determined."""
    avail = None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    for p in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(p) as f:
                s = f.read().strip()
            if s != 'max':
                lim = int(s)
                if 0 < lim < 1 << 50 and (avail is None or lim < avail):
                    avail = lim
        except (OSError, ValueError):
            pass
    return avail


def memory_budget_bytes(budget_mb=None):
    """Explicit --memory-budget (MB) or 50% of the available memory (4 GB when unknown)."""
    if budget_mb:
        return int(float(budget_mb) * 1048576)
    avail = available_memory_bytes()
    return int(avail * 0.5) if avail else 4 << 30


def add_common_args(ap):
    ap.add_argument('--memory-budget', type=float, default=None, metavar='MB',
                    help='peak memory the tool may use (default: 50%% of the available memory)')
    ap.add_argument('--temp-dir', default=None,
                    help=f'folder for temporary files (default $PIPELINE_TEMP_DIR or {DEFAULT_TEMP_DIR})')


# ---------------------------------------------------------------------------- temp work dir

class TempWorkDir:
    """
    One folder per job under the temp root. Removed when the `with` block ends, on an exception, at
    interpreter exit and on SIGTERM (/stop_process/ kills the process group with SIGTERM).
    """

    def __init__(self, root=None, prefix='job_', need_bytes=0):
        root = root or os.environ.get('PIPELINE_TEMP_DIR') or DEFAULT_TEMP_DIR
        os.makedirs(root, exist_ok=True)
        if need_bytes:
            free = shutil.disk_usage(root).free
            if free < need_bytes * 1.05:
                raise RuntimeError(
                    f'Not enough free space in the temporary folder {root}: need ~{need_bytes / 1e9:.1f} GB, '
                    f'{free / 1e9:.1f} GB free (set PIPELINE_TEMP_DIR to a bigger volume)')
        self.path = tempfile.mkdtemp(prefix=f'{prefix}{os.getpid()}_', dir=root)
        self._prev = None

    def _cleanup(self):
        if self.path:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None

    def _on_term(self, signum, frame):
        self._cleanup()
        if callable(self._prev):        # chain to the handler installed before (removes tmp outputs)
            self._prev(signum, frame)
        sys.stdout.flush()
        os._exit(128 + signum)

    def __enter__(self):
        atexit.register(self._cleanup)
        try:
            self._prev = signal.signal(signal.SIGTERM, self._on_term)
        except ValueError:  # not in the main thread
            self._prev = None
        return self

    def __exit__(self, *exc):
        self._cleanup()
        if self._prev is not None:
            signal.signal(signal.SIGTERM, self._prev)
        return False

    def file(self, name):
        return os.path.join(self.path, name)


def install_term_cleanup(paths_fn):
    """SIGTERM handler for tools without a temp folder: removes `paths_fn()` (files) then exits."""
    def handler(signum, frame):
        for p in paths_fn():
            try:
                os.remove(p)
            except OSError:
                pass
        sys.stdout.flush()
        os._exit(128 + signum)
    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:
        pass


# ---------------------------------------------------------------------------- LAS by blocks

class LasFile:
    """
    Header, VLRs and Extra Bytes of an uncompressed LAS 1.x plus a sequential block reader.

    `fields` maps a name to (numpy dtype, byte offset inside the record) for the Extra Bytes and the
    standard fields the pipeline needs (X Y Z intensity red green blue).
    """

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            head = f.read(375)
            if head[:4] != b'LASF':
                raise ValueError(f'Not a LAS file: {path}')
            self.version_minor = head[25]
            self.header_size, = struct.unpack_from('<H', head, 94)
            self.data_offset, = struct.unpack_from('<I', head, 96)
            self.num_vlrs, = struct.unpack_from('<I', head, 100)
            fmt = head[104]
            if fmt & 0xC0:
                raise ValueError('Compressed LAS (LAZ) is not supported')
            self.fmt = fmt & 0x3F
            self.record_len, = struct.unpack_from('<H', head, 105)
            self.n, = struct.unpack_from('<I', head, 107)
            if self.n == 0 and self.version_minor >= 4 and self.header_size >= 375:
                self.n, = struct.unpack_from('<Q', head, 247)
            self.scale = struct.unpack_from('<3d', head, 131)
            self.offset = struct.unpack_from('<3d', head, 155)
            base = BASE_RECORD_SIZES.get(self.fmt)
            if base is None:
                raise ValueError(f'Unsupported LAS point format {self.fmt}')
            self.base_size = base
            f.seek(0)
            self.header_bytes = f.read(self.data_offset)   # header + VLRs (+ anything up to the points)

        self.fields = {'X': ('<i4', 0), 'Y': ('<i4', 4), 'Z': ('<i4', 8), 'intensity': ('<u2', 12)}
        rgb = RGB_OFFSETS.get(self.fmt)
        if rgb is not None:
            for i, c in enumerate(('red', 'green', 'blue')):
                self.fields[c] = ('<u2', rgb + 2 * i)
        # Other plain fields (names as in laspy). Bit-packed ones (return_number, flags) are not exposed.
        if self.fmt <= 5:
            self.fields.update({'classification': ('<u1', 15), 'scan_angle_rank': ('<i1', 16),
                                'user_data': ('<u1', 17), 'point_source_id': ('<u2', 18)})
            if self.fmt in (1, 3, 4, 5):
                self.fields['gps_time'] = ('<f8', 20)
        else:
            self.fields.update({'classification': ('<u1', 16), 'user_data': ('<u1', 17),
                                'scan_angle': ('<i2', 18), 'point_source_id': ('<u2', 20),
                                'gps_time': ('<f8', 22)})
        self.extra_names = []
        self.extra_vlr = None            # (position of the VLR header, record length, absolute descr. offset)
        pos = self.header_size
        hb = self.header_bytes
        for _ in range(self.num_vlrs):
            if pos + 54 > len(hb):
                break
            user_id = hb[pos + 2:pos + 18].rstrip(b'\x00').decode('latin-1')
            record_id, = struct.unpack_from('<H', hb, pos + 18)
            length, = struct.unpack_from('<H', hb, pos + 20)
            body = pos + 54
            if user_id.startswith('LASF_Spec') and record_id == 4:
                self.extra_vlr = (pos, length)
                cur = base
                for i in range(length // 192):
                    rec = hb[body + i * 192: body + (i + 1) * 192]
                    dtype, options = rec[2], rec[3]
                    name = rec[4:36].split(b'\x00')[0].decode('latin-1').strip()
                    size = options if dtype == 0 else EXTRA_SIZES.get(dtype, 0)
                    if dtype in EXTRA_DTYPES and name and name not in self.fields:
                        self.fields[name] = (EXTRA_DTYPES[dtype], cur)
                        self.extra_names.append(name)
                    cur += size
                self.extra_end = cur
            pos = body + length
        self.vlr_end = pos
        self.evlr_count = 0
        if self.version_minor >= 4 and self.header_size >= 375:
            self.evlr_count, = struct.unpack_from('<I', head, 243)
        self.size = os.path.getsize(path)
        if self.n > (self.size - self.data_offset) // max(1, self.record_len):
            raise ValueError('LAS file is truncated')

    # -- dtype of a block with only the requested fields (zero-copy views over the raw buffer)
    def block_dtype(self, names):
        names = list(dict.fromkeys(names))
        missing = [n for n in names if n not in self.fields]
        if missing:
            raise KeyError(missing)
        return np.dtype({'names': names,
                         'formats': [self.fields[n][0] for n in names],
                         'offsets': [self.fields[n][1] for n in names],
                         'itemsize': self.record_len})

    def block_rows_for(self, budget_bytes, per_row_extra=0, lo=1 << 16, hi=4 << 20):
        """Rows per block so that raw buffer + `per_row_extra` bytes per row stay within the budget."""
        rows = int(budget_bytes // max(1, self.record_len + per_row_extra))
        return max(lo, min(hi, rows))

    def raw_blocks(self, rows_per_block, start=0, stop=None, advise=True):
        """Yields (first_row, uint8 array (m, record_len)). Sequential pread, cache dropped behind the cursor."""
        stop = self.n if stop is None else min(stop, self.n)
        rl = self.record_len
        fd = os.open(self.path, os.O_RDONLY)
        try:
            if advise and hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            row = start
            while row < stop:
                m = min(rows_per_block, stop - row)
                buf = np.empty((m, rl), dtype=np.uint8)
                mv = memoryview(buf).cast('B')
                off = self.data_offset + row * rl
                got = 0
                while got < m * rl:
                    k = os.preadv(fd, [mv[got:]], off + got)
                    if k <= 0:
                        raise IOError('unexpected end of file')
                    got += k
                yield row, buf
                if advise and hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(fd, off, m * rl, os.POSIX_FADV_DONTNEED)   # this block only (clean pages)
                row += m
        finally:
            os.close(fd)

    def blocks(self, names, rows_per_block, **kw):
        """Yields (first_row, structured array exposing only `names`)."""
        dt = self.block_dtype(names)
        for row, buf in self.raw_blocks(rows_per_block, **kw):
            yield row, np.ndarray(shape=(buf.shape[0],), dtype=dt, buffer=buf)


def drop_cache(fd_or_path, offset=0, length=0):
    """Best effort: tell the kernel the (clean, already written) pages of a file are not needed."""
    if not hasattr(os, 'posix_fadvise'):
        return
    fd = os.open(fd_or_path, os.O_RDONLY) if isinstance(fd_or_path, str) else fd_or_path
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        if isinstance(fd_or_path, str):
            os.close(fd)


_LIBC = None


def _sync_file_range(fd, offset, nbytes, flags):
    """Linux sync_file_range(2) through libc (not exposed by the os module); returns False if unavailable."""
    global _LIBC
    try:
        if _LIBC is None:
            import ctypes
            _LIBC = ctypes.CDLL(None, use_errno=True)
            _LIBC.sync_file_range.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_uint]
        return _LIBC.sync_file_range(fd, offset, nbytes, flags) == 0
    except (OSError, AttributeError):
        return False


SYNC_WAIT_BEFORE, SYNC_WRITE, SYNC_WAIT_AFTER = 1, 2, 4


class SequentialWriter:
    """
    Append-only file writer that keeps the page cache small (cgroup friendly): behind the cursor, windows
    of `window` bytes are first sent to disk asynchronously (sync_file_range WRITE) and one window later
    waited for and dropped (posix_fadvise DONTNEED; dirty pages could not be dropped). Small files never
    reach the first window, so they cost nothing.
    """

    def __init__(self, path, window=64 << 20):
        self.path = path
        self.f = open(path, 'wb', buffering=1 << 20)
        self.written = 0
        self._kicked = 0       # [_dropped, _kicked) was handed to the disk
        self._dropped = 0      # [0, _dropped) was waited for and dropped from the cache
        self._win = window

    def write(self, data):
        self.f.write(data)
        self.written += len(data)
        if self.written - self._kicked >= self._win:
            self._advance()

    def _advance(self):
        self.f.flush()
        fd = self.f.fileno()
        n = self._win
        if not _sync_file_range(fd, self._kicked, n, SYNC_WRITE):
            os.fdatasync(fd)
        self._kicked += n
        if self._kicked - self._dropped >= 4 * n:
            _sync_file_range(fd, self._dropped, n, SYNC_WAIT_BEFORE | SYNC_WRITE | SYNC_WAIT_AFTER)
            os.posix_fadvise(fd, self._dropped, n, os.POSIX_FADV_DONTNEED)
            self._dropped += n

    def close(self):
        self.f.flush()
        fd = self.f.fileno()
        if self._dropped:     # a big file: do not leave the tail dirty in the cache either
            _sync_file_range(fd, self._dropped, 0, SYNC_WRITE)
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def log_phase(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def prefetched(iterator, depth=2):
    """Runs `iterator` in a thread so that reading (GIL-free pread) overlaps with the caller's work."""
    import queue
    import threading
    q = queue.Queue(maxsize=depth)
    stop = threading.Event()
    DONE = object()

    def run():
        try:
            for item in iterator:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(DONE)
        except BaseException as e:  # hand the error to the consumer
            try:
                q.put(e, timeout=5)
            except queue.Full:
                pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=5)


def build_features(blk, features, las, out):
    """
    Fills out (m, nf) float32 with the `features` of a block (structured array from LasFile.blocks).
    X/Y/Z are (integer * scale + offset) - offset in float64, as laspy gives them minus the LAS offset; the
    classification of formats 0-5 keeps its 5 class bits (the other 3 are flags).
    """
    for j, name in enumerate(features):
        if name in ('X', 'Y', 'Z'):
            k = 'XYZ'.index(name)
            world = np.asarray(blk[name], dtype=np.float64) * las.scale[k] + las.offset[k]
            out[:, j] = world - las.offset[k]
        elif name == 'classification' and las.fmt <= 5:
            out[:, j] = np.asarray(blk[name]) & 0x1F
        else:
            out[:, j] = np.asarray(blk[name], dtype=np.float64)


class Progress:
    """Prints '[progress] <phase> NN%' lines (every 10 %) for the backend log."""

    def __init__(self, phase, total):
        self.phase, self.total, self._last = phase, max(1, total), -1

    def update(self, done):
        pct = int(100 * done / self.total) // 10 * 10
        if pct != self._last:
            self._last = pct
            print(f'[progress] {self.phase} {pct}%', flush=True)
