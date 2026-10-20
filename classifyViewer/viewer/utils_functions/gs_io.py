"""Readers for Gaussian splat sources, all converted to one in-memory `GaussianSet`.

Supported inputs:
  * 3DGS PLY (binary little endian, scale_0..2)
  * 2DGS PLY (binary little endian, scale_0..1: each splat is an oriented disc)
  * SOG v2 (PlayCanvas splat-transform, ZIP with meta.json and WebP textures)

Values are kept as the source stores them: scales are log values, opacity is a logit,
colour is the DC coefficient, SH are the `f_rest` coefficients in channel-major order
(R coefficients, then G, then B). The only change is quaternion normalisation, so that
every splat has a unit rotation (rot_0..3 = w, x, y, z).

SPZ is not supported yet.
"""
import io
import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

# Number of f_rest values per splat -> SH degree (3 * ((deg + 1)^2 - 1)).
SH_REST_TO_DEGREE = {0: 0, 9: 1, 24: 2, 45: 3}
# SOG shN.bands -> coefficients per colour channel.
SOG_SH_COEFFS = [0, 3, 8, 15]
# Smallest-three quaternion packing: the three stored components per largest-component index.
_QUAT_IDX = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_SQRT2 = np.sqrt(2.0)
# Opacity clamp used by splat-transform when it inverts the sigmoid.
_SIGMOID_EPS = 1e-6

_PLY_TYPES = {
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
}


@dataclass
class GaussianSet:
    """Splats in source order. Row i is splat i of the source file (its stable ID)."""
    kind: str                 # '3dgs' or '2dgs'
    source: str               # '3dgs-ply', '2dgs-ply' or 'sog'
    xyz: np.ndarray           # (N, 3) float32
    rot: np.ndarray           # (N, 4) float32, unit quaternion w, x, y, z
    log_scale: np.ndarray     # (N, 3) float32 for 3DGS, (N, 2) for 2DGS
    opacity: np.ndarray       # (N,) float32, logit
    f_dc: np.ndarray          # (N, 3) float32
    f_rest: np.ndarray        # (N, 3 * (K - 1)) float32, channel-major, (N, 0) if sh_degree 0
    sh_degree: int
    stats: dict = field(default_factory=dict)  # raw quaternion norms, counts of odd values

    def __len__(self):
        return len(self.xyz)


# ----------------------------------------------------------------------------- PLY

def _read_ply_vertex(path):
    """Return (memmap of the vertex records, property names) of a binary PLY file."""
    with open(path, 'rb') as f:
        head = f.read(1 << 16)
    end = head.find(b'end_header')
    if end < 0:
        raise ValueError(f'{path}: not a PLY file, or the header is longer than 64 KiB')
    header_end = head.find(b'\n', end) + 1
    lines = head[:end].decode('ascii', 'replace').splitlines()
    if not lines or lines[0].strip() != 'ply':
        raise ValueError(f'{path}: missing "ply" magic')

    fmt, elements = None, []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            fmt = parts[1]
        elif parts[0] == 'element':
            elements.append({'name': parts[1], 'count': int(parts[2]), 'props': []})
        elif parts[0] == 'property':
            if parts[1] == 'list':
                raise ValueError(f'{path}: list properties are not supported')
            elements[-1]['props'].append((parts[2], _PLY_TYPES[parts[1]]))

    if fmt != 'binary_little_endian':
        raise ValueError(f'{path}: format {fmt} not supported (binary_little_endian only)')
    if len(elements) != 1 or elements[0]['name'] != 'vertex':
        raise ValueError(f'{path}: expected a single "vertex" element, found {[e["name"] for e in elements]}')

    vertex = elements[0]
    dtype = np.dtype([(name, t) for name, t in vertex['props']])
    count = vertex['count']
    size = os.path.getsize(path)
    if size - header_end != count * dtype.itemsize:
        raise ValueError(f'{path}: size mismatch, expected {header_end + count * dtype.itemsize} bytes, found {size}')
    data = np.memmap(path, dtype=dtype, mode='r', offset=header_end, shape=(count,))
    return data, [name for name, _ in vertex['props']]


def _f32(arr):
    return np.array(arr, dtype=np.float32)


def read_ply(path):
    """Read a 3DGS or 2DGS PLY file. The kind is 2DGS when the file has no scale_2."""
    data, names = _read_ply_vertex(path)
    if 'scale_1' not in names:
        raise ValueError(f'{path}: no scale_0/scale_1 properties, not a Gaussian splat file')
    kind = '2dgs' if 'scale_2' not in names else '3dgs'
    scale_names = ['scale_0', 'scale_1'] + (['scale_2'] if kind == '3dgs' else [])

    rest_names = sorted((n for n in names if n.startswith('f_rest_')), key=lambda n: int(n[7:]))
    if len(rest_names) not in SH_REST_TO_DEGREE:
        raise ValueError(f'{path}: {len(rest_names)} f_rest properties, expected one of {sorted(SH_REST_TO_DEGREE)}')
    sh_degree = SH_REST_TO_DEGREE[len(rest_names)]

    xyz = np.stack([_f32(data[c]) for c in ('x', 'y', 'z')], axis=1)
    rot_raw = np.stack([_f64(data[f'rot_{i}']) for i in range(4)], axis=1)
    rot, raw_norm = _normalise_quat(rot_raw)
    log_scale = np.stack([_f32(data[n]) for n in scale_names], axis=1)
    opacity = _f32(data['opacity'])
    f_dc = np.stack([_f32(data[f'f_dc_{i}']) for i in range(3)], axis=1)
    if rest_names:
        f_rest = np.stack([_f32(data[n]) for n in rest_names], axis=1)
    else:
        f_rest = np.zeros((len(data), 0), dtype=np.float32)

    normals_nonzero = int(np.count_nonzero(np.any(
        np.stack([_f32(data[c]) for c in ('nx', 'ny', 'nz')], axis=1) != 0, axis=1))) if 'nx' in names else 0
    return GaussianSet(
        kind=kind, source=f'{kind}-ply', xyz=xyz, rot=rot.astype(np.float32), log_scale=log_scale,
        opacity=opacity, f_dc=f_dc, f_rest=f_rest, sh_degree=sh_degree,
        stats={'quat_norm_raw_min': float(raw_norm.min()), 'quat_norm_raw_max': float(raw_norm.max()),
               'normals_nonzero': normals_nonzero, 'ply_properties': names},
    )


def _f64(arr):
    return np.array(arr, dtype=np.float64)


# ----------------------------------------------------------------------------- SOG

def _pil_image():
    """Pillow is needed only for SOG (WebP textures): PLY files work without it."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('SOG import needs Pillow (pip install pillow)') from e
    return Image


def _webp_rgba(blob, count):
    """Decode a WebP texture into an (count, 4) uint8 array, one row per splat (row-major texels)."""
    img = _pil_image().open(io.BytesIO(blob)).convert('RGBA')
    flat = np.asarray(img, dtype=np.uint8).reshape(-1, 4)
    if flat.shape[0] < count:
        raise ValueError(f'SOG texture has {flat.shape[0]} texels, {count} needed')
    return flat[:count]


def _inv_log(v):
    """Inverse of splat-transform's logTransform: sign(v) * (exp(|v|) - 1)."""
    return np.sign(v) * np.expm1(np.abs(v))


def _normalise_quat(q):
    """Return (unit quaternions float64, raw norms). Zero quaternions become the identity."""
    norm = np.linalg.norm(q, axis=1)
    out = np.where(norm[:, None] > 0, q / np.where(norm[:, None] > 0, norm[:, None], 1.0), [1.0, 0.0, 0.0, 0.0])
    return out, norm


def read_sog(path):
    """Read a SOG v2 file (PlayCanvas splat-transform). Mirrors splat-transform's V2 reader."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read('meta.json'))
        if meta.get('version') != 2:
            raise ValueError(f'{path}: SOG version {meta.get("version")} not supported (v2 only)')
        count = int(meta['count'])
        tex = lambda name: _webp_rgba(z.read(name), count)  # noqa: E731

        # Position: 16 bit per axis split in two textures, lerp between mins/maxs, then inverse log.
        lo = tex(meta['means']['files'][0]).astype(np.uint16)
        hi = tex(meta['means']['files'][1]).astype(np.uint16)
        mins = np.asarray(meta['means']['mins'], dtype=np.float64)
        maxs = np.asarray(meta['means']['maxs'], dtype=np.float64)
        span = np.where(maxs - mins == 0, 1.0, maxs - mins)
        v16 = (lo[:, :3] | (hi[:, :3] << 8)).astype(np.float64)
        xyz = _inv_log(mins + span * (v16 / 65535.0)).astype(np.float32)

        # Rotation: smallest-three packing, alpha is the largest-component tag (252..255).
        q = tex(meta['quats']['files'][0])
        tag = q[:, 3].astype(np.int64)
        valid = (tag >= 252) & (tag <= 255)
        max_comp = np.clip(tag - 252, 0, 3)
        comp = (q[:, :3].astype(np.float64) / 255.0 * 2.0 - 1.0) / _SQRT2
        largest = np.sqrt(np.maximum(0.0, 1.0 - np.sum(comp * comp, axis=1)))
        rot_all = np.zeros((count, 4), dtype=np.float64)
        for m in range(4):
            sel = valid & (max_comp == m)
            rot_all[np.ix_(sel.nonzero()[0], _QUAT_IDX[m])] = comp[sel]
            rot_all[sel, m] = largest[sel]
        rot_all[~valid] = [1.0, 0.0, 0.0, 0.0]
        rot, raw_norm = _normalise_quat(rot_all)

        # Scale: one byte per axis into a shared 256-entry codebook (already log values).
        scale_code = np.asarray(meta['scales']['codebook'], dtype=np.float64)
        s = tex(meta['scales']['files'][0])
        log_scale = scale_code[s[:, :3]].astype(np.float32)

        # Colour: DC codebook for RGB, alpha is the opacity as a sigmoid byte -> logit.
        c0 = tex(meta['sh0']['files'][0])
        dc_code = np.asarray(meta['sh0']['codebook'], dtype=np.float64)
        f_dc = dc_code[c0[:, :3]].astype(np.float32)
        eps = np.clip(c0[:, 3].astype(np.float64) / 255.0, _SIGMOID_EPS, 1.0 - _SIGMOID_EPS)
        opacity = np.log(eps / (1.0 - eps)).astype(np.float32)

        # Higher-order SH: label -> palette slot -> per-coefficient byte -> codebook.
        sh_degree = 0
        f_rest = np.zeros((count, 0), dtype=np.float32)
        shn = meta.get('shN')
        if shn and SOG_SH_COEFFS[shn['bands']] > 0:
            k = SOG_SH_COEFFS[shn['bands']]
            sh_degree = shn['bands']
            cen = np.asarray(_pil_image().open(io.BytesIO(z.read(shn['files'][0]))).convert('RGBA'), dtype=np.uint8)
            c_h, c_w = cen.shape[0], cen.shape[1]
            if c_w != 64 * k:
                raise ValueError(f'SOG shN centroids width {c_w} does not match {64 * k} for {shn["bands"]} bands')
            cen_flat = cen.reshape(-1, 4)
            lab = tex(shn['files'][1]).astype(np.int64)
            label = lab[:, 0] | (lab[:, 1] << 8)
            palette_ok = label < int(shn['count'])
            cy = label // 64
            cx0 = (label % 64) * k
            row_ok = palette_ok & (cy < c_h)
            shn_codebook = np.asarray(shn['codebook'], dtype=np.float64)
            f_rest = np.zeros((count, 3 * k), dtype=np.float32)
            for j in range(k):
                ok = row_ok & (cx0 + j < c_w)
                # Texels outside the palette read byte 0, as splat-transform does (codebook[0]).
                px = np.where(ok[:, None], cen_flat[np.where(ok, cy * c_w + cx0 + j, 0)], 0)
                for ch in range(3):
                    f_rest[:, ch * k + j] = shn_codebook[px[:, ch]]
            f_rest[~palette_ok] = 0.0

    return GaussianSet(
        kind='3dgs', source='sog', xyz=xyz, rot=rot.astype(np.float32), log_scale=log_scale,
        opacity=opacity, f_dc=f_dc, f_rest=f_rest, sh_degree=sh_degree,
        stats={'quat_norm_raw_min': float(raw_norm.min()), 'quat_norm_raw_max': float(raw_norm.max()),
               'invalid_quat_tags': int(np.count_nonzero(~valid))},
    )


# ----------------------------------------------------------------------------- dispatch

def read_source(path):
    """Read any supported source file, chosen by extension."""
    lower = path.lower()
    if lower.endswith('.ply'):
        return read_ply(path)
    if lower.endswith('.sog'):
        return read_sog(path)
    raise ValueError(f'{path}: unsupported extension (3DGS PLY, 2DGS PLY and SOG are supported)')


def describe(gs):
    """Summary statistics for the report and for sanity checks."""
    scale_max = gs.log_scale.max(axis=1)
    opacity = 1.0 / (1.0 + np.exp(-gs.opacity.astype(np.float64)))
    return {
        'count': len(gs),
        'kind': gs.kind,
        'source': gs.source,
        'sh_degree': gs.sh_degree,
        'nonfinite': int(np.count_nonzero(~np.isfinite(gs.xyz)) + np.count_nonzero(~np.isfinite(gs.log_scale))
                         + np.count_nonzero(~np.isfinite(gs.opacity)) + np.count_nonzero(~np.isfinite(gs.rot))),
        'xyz_min': gs.xyz.min(axis=0).tolist(),
        'xyz_max': gs.xyz.max(axis=0).tolist(),
        'log_scale_min': float(gs.log_scale.min()),
        'log_scale_max': float(gs.log_scale.max()),
        'splats_scale_below_e10': int(np.count_nonzero(scale_max < -10.0)),
        'opacity_min': float(opacity.min()),
        'opacity_max': float(opacity.max()),
        **gs.stats,
    }
