"""Internal Gaussian splat format `gs` (formatVersion 1) and the 3DGS/2DGS PLY writer.

Layout of a `gs` folder:

    meta.json     format description, chunks (Morton order), column list
    xyz.bin       float32 (N, 3)   absolute position, source values
    rot.bin       float32 (N, 4)   unit quaternion w, x, y, z
    scale.bin     float32 (N, S)   log scales, S = 3 (3DGS) or 2 (2DGS)
    opacity.bin   float32 (N,)     logit
    dc.bin        float32 (N, 3)   DC colour
    sh.bin        float32 (N, 3*(K-1))  SH rest, channel-major (only if sh_degree > 0)
    id.bin        uint32 (N,)      source index of each row (stable splat ID)

Rows are in Morton order, grouped in chunks of at most MAX_CHUNK splats. `id.bin` maps a row back
to the source splat, so the file can be restored to source order and edits keyed by ID stay valid.
All columns are exact copies of the source values: `gs -> PLY` is bit-exact (quaternions are
normalised once at import, see gs_io.py).
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

import gs_quant
from gs_io import GaussianSet

FORMAT = 'gs'
FORMAT_VERSION = 2          # quantised (default); 1 is the exact float32 reference
MAX_CHUNK = 250_000
LOD_SEED = 20261014      # seed of the shuffle inside each chunk (chunk id is added)
ORDER_MORTON_RANDOM = 'morton-chunk-random'   # rows Morton-ordered by chunk, shuffled inside each chunk
PLY_BLOCK = 262_144  # rows per write call, bounds the memory of the PLY writer
_BIT_MASK = np.uint64(0x1FFFFF)
_DISK_DTYPE = {'uint8': 'u1', 'uint16': '<u2', 'uint32': '<u4', 'float32': '<f4'}


def _spread_bits(x):
    """Spread the 21 low bits of x so that they occupy every third bit (3D Morton code)."""
    x = x & _BIT_MASK
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_order(xyz):
    """Stable permutation that sorts the points along a 3D Morton curve (ties by source index)."""
    lo = xyz.min(axis=0).astype(np.float64)
    span = xyz.max(axis=0).astype(np.float64) - lo
    span[span == 0] = 1.0
    q = np.floor((xyz - lo) / span * float((1 << 21) - 1)).astype(np.uint64)
    code = _spread_bits(q[:, 0]) | (_spread_bits(q[:, 1]) << np.uint64(1)) | (_spread_bits(q[:, 2]) << np.uint64(2))
    return np.lexsort((np.arange(len(xyz)), code))


def _columns(gs):
    """Columns of a GaussianSet: name -> (array, components). Arrays are in source order."""
    cols = {
        'xyz': (gs.xyz, 3),
        'rot': (gs.rot, 4),
        'scale': (gs.log_scale, gs.log_scale.shape[1]),
        'opacity': (gs.opacity, 1),
        'dc': (gs.f_dc, 3),
    }
    if gs.sh_degree > 0:
        cols['sh'] = (gs.f_rest, gs.f_rest.shape[1])
    return cols


def write_gs(gs, out_dir, version=FORMAT_VERSION, sh_bits=8):
    """Write a GaussianSet as a `gs` folder. version 2 quantises the columns (see gs_quant.py),
    version 1 writes float32. Returns the metadata dictionary."""
    if version not in (1, 2):
        raise ValueError(f'unknown gs format version {version}')
    os.makedirs(out_dir, exist_ok=True)
    n = len(gs)
    if n >= 2 ** 32:
        raise ValueError('too many splats for uint32 IDs')
    order = morton_order(gs.xyz)

    chunks = []
    for start in range(0, n, MAX_CHUNK):
        rows = order[start:start + MAX_CHUNK]
        pts = gs.xyz[rows]
        chunks.append({
            'id': len(chunks), 'start': start, 'count': len(rows),
            'min': pts.min(axis=0).astype(float).tolist(), 'max': pts.max(axis=0).astype(float).tolist(),
        })
    # LOD: inside each chunk the rows are shuffled (fixed seed per chunk). Then any prefix of a chunk is a uniform
    # sample of it, so the renderer can draw a fraction of the chunk and compensate the splats (see gs-splat.js).
    if version == 2:
        for ch in chunks:
            rng = np.random.default_rng(LOD_SEED + ch['id'])
            s, e = ch['start'], ch['start'] + ch['count']
            order[s:e] = order[s:e][rng.permutation(e - s)]

    columns = {}
    if version == 1:
        for name, (arr, comps) in _columns(gs).items():
            file_name = f'{name}.bin'
            arr[order].astype('<f4').tofile(os.path.join(out_dir, file_name))
            columns[name] = {'file': file_name, 'dtype': 'float32', 'components': int(comps)}
    else:
        encoded, chunks = gs_quant.encode(gs, order, chunks, sh_bits=sh_bits)
        for name, (arr, dtype, comps) in encoded.items():
            file_name = f'{name}.bin'
            arr.astype(_DISK_DTYPE[dtype]).tofile(os.path.join(out_dir, file_name))
            columns[name] = {'file': file_name, 'dtype': dtype, 'components': int(comps)}
        columns['xyz']['encoding'] = 'uint24-le per axis, 3 bytes each'
    order.astype('<u4').tofile(os.path.join(out_dir, 'id.bin'))
    columns['id'] = {'file': 'id.bin', 'dtype': 'uint32', 'components': 1}

    meta = {
        'format': FORMAT,
        'formatVersion': version,
        'kind': gs.kind,
        'count': n,
        'shDegree': gs.sh_degree,
        'source': {'format': gs.source, 'count': n},
        'order': ORDER_MORTON_RANDOM if version == 2 else 'morton',
        'maxChunk': MAX_CHUNK,
        'chunks': chunks,
        'columns': columns,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta


def read_gs(gs_dir, source_order=True):
    """Read a `gs` folder back. With source_order=True the rows are restored to source order."""
    with open(f'{gs_dir}/meta.json', encoding='utf-8') as f:
        meta = json.load(f)
    version = meta.get('formatVersion')
    if meta.get('format') != FORMAT or version not in (1, 2):
        raise ValueError(f'{gs_dir}: not a {FORMAT} folder (version 1 or 2)')
    n = meta['count']
    cols = meta['columns']

    def load(name):
        c = cols[name]
        arr = np.fromfile(f"{gs_dir}/{c['file']}", dtype=_DISK_DTYPE[c['dtype']])
        return arr.reshape(n, c['components']) if c['components'] > 1 else arr.reshape(n)

    ids = load('id').astype(np.int64)
    inv = np.empty(n, dtype=np.int64)
    inv[ids] = np.arange(n)

    names = ['xyz', 'rot', 'scale', 'opacity', 'dc'] + (['sh'] if 'sh' in cols else [])
    raw = {name: load(name) for name in names}
    if version == 1:
        arrays = {name: raw[name].astype(np.float32) for name in names}
    else:
        k_scale = cols['scale']['components']
        arrays = gs_quant.decode(raw, meta['chunks'], k_scale)
        arrays = {'xyz': arrays['xyz'], 'rot': arrays['rot'], 'scale': arrays['log_scale'],
                  'opacity': arrays['opacity'], 'dc': arrays['f_dc'], 'sh': arrays['f_rest']}
        if 'sh' not in cols:
            arrays.pop('sh')
    if source_order:
        arrays = {name: arr[inv] for name, arr in arrays.items()}
    return GaussianSet(
        kind=meta['kind'], source=meta['source']['format'],
        xyz=arrays['xyz'], rot=arrays['rot'], log_scale=arrays['scale'], opacity=arrays['opacity'],
        f_dc=arrays['dc'], f_rest=arrays.get('sh', np.zeros((n, 0), dtype=np.float32)),
        sh_degree=meta['shDegree'],
    )


def write_ply(gs, out_path):
    """Write a standard binary PLY (the layout of the INRIA 3DGS and 2DGS files).

    Properties: x y z, nx ny nz (zero), f_dc_0..2, f_rest_0..K-1, opacity, scale_*, rot_0..3.
    A 2DGS set keeps its two scales. Normals are written as zero, as the trainers do.
    """
    n = len(gs)
    k_rest = gs.f_rest.shape[1]
    scale_n = gs.log_scale.shape[1]
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(k_rest)] + ['opacity']
             + [f'scale_{i}' for i in range(scale_n)] + [f'rot_{i}' for i in range(4)])
    header = ['ply', 'format binary_little_endian 1.0', 'comment written by gs_format.py',
              f'element vertex {n}']
    header += [f'property float {name}' for name in names]
    header.append('end_header')
    with open(out_path, 'wb') as f:
        f.write(('\n'.join(header) + '\n').encode('ascii'))
        for start in range(0, n, PLY_BLOCK):
            end = min(n, start + PLY_BLOCK)
            m = end - start
            block = np.zeros((m, len(names)), dtype='<f4')
            block[:, 0:3] = gs.xyz[start:end]
            block[:, 6:9] = gs.f_dc[start:end]
            col = 9
            block[:, col:col + k_rest] = gs.f_rest[start:end]
            col += k_rest
            block[:, col] = gs.opacity[start:end]
            col += 1
            block[:, col:col + scale_n] = gs.log_scale[start:end]
            col += scale_n
            block[:, col:col + 4] = gs.rot[start:end]
            f.write(block.tobytes())
    return names
