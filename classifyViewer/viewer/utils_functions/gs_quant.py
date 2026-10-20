"""Quantisation for the `gs` format v2, and the error report used to choose the bit depths.

Every field is quantised per chunk, with the range (min, max) of the chunk stored in meta.json:

  xyz      uint16 per axis, inside the chunk's tight AABB
  rot      smallest-three: three uint16 components in [-1/sqrt2, 1/sqrt2] + uint16 index of the dropped one
  scale    uint16 per axis, linear in the log-scale range of the chunk
  opacity  uint16, linear in the logit range of the chunk
  dc       uint16 per channel, linear in the DC range of the chunk
  sh       uint8 (or uint16) per coefficient, linear in the range of that coefficient in the chunk

The error is measured against the exact v1 values (gs_io output). The report gives, for each field, the
maximum error and, for positions, the error relative to the splat size.
"""
import numpy as np

_QUAT_IDX = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_SQRT2 = np.sqrt(2.0)


def _maxq(bits):
    return float((1 << bits) - 1)


def _range(vals):
    """Per-column (lo, hi) of a 2-D float64 array. Empty chunks are not passed in."""
    return vals.min(axis=0), vals.max(axis=0)


def quantise_linear(vals, lo, hi, bits):
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    q = np.rint((vals - lo) / span * _maxq(bits))
    dtype = np.uint8 if bits <= 8 else np.uint16 if bits <= 16 else np.uint32
    return np.clip(q, 0, _maxq(bits)).astype(dtype)


def pack24(q):
    """(N, K) uint32 values below 2^24 -> (N, 3K) uint8, three little-endian bytes per value."""
    b = np.stack([q & 0xFF, (q >> 8) & 0xFF, (q >> 16) & 0xFF], axis=-1).astype(np.uint8)
    return b.reshape(len(q), -1)


def unpack24(b):
    """Inverse of pack24: (N, 3K) uint8 -> (N, K) uint32."""
    b = b.reshape(len(b), -1, 3).astype(np.uint32)
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)


def dequantise_linear(q, lo, hi, bits):
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return lo + span * (q.astype(np.float64) / _maxq(bits))


def encode_quat(q):
    """Smallest-three packing of unit quaternions (w, x, y, z) into 3 x uint16 + index (uint16)."""
    rows = np.arange(len(q))
    m = np.argmax(np.abs(q), axis=1)
    s = q * np.sign(q[rows, m])[:, None]          # largest component positive, so it is not stored
    others = s[rows[:, None], _QUAT_IDX[m]]        # (N, 3) in [-1/sqrt2, 1/sqrt2]
    packed = np.rint((others * _SQRT2 + 1.0) / 2.0 * 65535.0)
    out = np.empty((len(q), 4), dtype=np.uint16)
    out[:, :3] = np.clip(packed, 0, 65535).astype(np.uint16)
    out[:, 3] = m.astype(np.uint16)
    return out


def decode_quat(p):
    rows = np.arange(len(p))
    m = p[:, 3].astype(np.int64)
    a = (p[:, :3].astype(np.float64) / 65535.0 * 2.0 - 1.0) / _SQRT2
    w = np.sqrt(np.maximum(0.0, 1.0 - np.sum(a * a, axis=1)))
    q = np.zeros((len(p), 4), dtype=np.float64)
    q[rows[:, None], _QUAT_IDX[m]] = a
    q[rows, m] = w
    return q / np.linalg.norm(q, axis=1)[:, None]


def encode(gs, order, chunks, sh_bits=8):
    """Quantise a GaussianSet whose rows are in `order` (Morton). Returns (columns, chunk params).

    columns: name -> (array in row order, dtype string, components)
    chunks:  list of dicts with the quantisation ranges, one per chunk (filled in place)
    """
    n = len(gs)
    xyz = gs.xyz[order].astype(np.float64)
    rot = gs.rot[order].astype(np.float64)
    scale = gs.log_scale[order].astype(np.float64)
    opacity = gs.opacity[order].astype(np.float64)
    dc = gs.f_dc[order].astype(np.float64)
    sh = gs.f_rest[order].astype(np.float64)
    k_scale = scale.shape[1]
    k_sh = sh.shape[1]

    out_xyz = np.zeros((n, 3), dtype=np.uint32)
    out_rot = np.zeros((n, 4), dtype=np.uint16)
    out_scale = np.zeros((n, k_scale), dtype=np.uint16)
    out_op = np.zeros(n, dtype=np.uint16)
    out_dc = np.zeros((n, 3), dtype=np.uint16)
    out_sh = np.zeros((n, k_sh), dtype=np.uint8 if sh_bits <= 8 else np.uint16)

    for ch in chunks:
        a, b = ch['start'], ch['start'] + ch['count']
        p_xyz = xyz[a:b]
        lo, hi = _range(p_xyz)
        ch['xyz'] = {'min': lo.tolist(), 'max': hi.tolist()}
        out_xyz[a:b] = quantise_linear(p_xyz, lo, hi, 24)

        out_rot[a:b] = encode_quat(rot[a:b])

        lo, hi = _range(scale[a:b])
        ch['scale'] = {'min': lo.tolist(), 'max': hi.tolist()}
        out_scale[a:b] = quantise_linear(scale[a:b], lo, hi, 16)

        lo, hi = opacity[a:b].min(), opacity[a:b].max()
        ch['opacity'] = {'min': float(lo), 'max': float(hi)}
        out_op[a:b] = quantise_linear(opacity[a:b], lo, hi, 16)

        lo, hi = _range(dc[a:b])
        ch['dc'] = {'min': lo.tolist(), 'max': hi.tolist()}
        out_dc[a:b] = quantise_linear(dc[a:b], lo, hi, 16)

        if k_sh:
            lo, hi = _range(sh[a:b])
            ch['sh'] = {'min': lo.tolist(), 'max': hi.tolist(), 'bits': sh_bits}
            out_sh[a:b] = quantise_linear(sh[a:b], lo, hi, sh_bits)

    cols = {
        'xyz': (pack24(out_xyz), 'uint8', 9),          # 3 x 24 bit, see pack24
        'rot': (out_rot, 'uint16', 4),
        'scale': (out_scale, 'uint16', k_scale),
        'opacity': (out_op, 'uint16', 1),
        'dc': (out_dc, 'uint16', 3),
    }
    if k_sh:
        cols['sh'] = (out_sh, 'uint8' if sh_bits <= 8 else 'uint16', k_sh)
    return cols, chunks


def decode(arrays, chunks, kind_k_scale, sh_bits=None):
    """Inverse of encode(): returns float32 arrays in row order (not source order).

    arrays: name -> ndarray with the stored dtype, rows in Morton order.
    """
    n = len(arrays['xyz'])
    xyz = np.zeros((n, 3), dtype=np.float64)
    rot = np.zeros((n, 4), dtype=np.float64)
    scale = np.zeros((n, kind_k_scale), dtype=np.float64)
    opacity = np.zeros(n, dtype=np.float64)
    dc = np.zeros((n, 3), dtype=np.float64)
    k_sh = arrays['sh'].shape[1] if 'sh' in arrays else 0
    sh = np.zeros((n, k_sh), dtype=np.float64)
    for ch in chunks:
        a, b = ch['start'], ch['start'] + ch['count']
        xyz[a:b] = dequantise_linear(unpack24(arrays['xyz'][a:b]), np.array(ch['xyz']['min']),
                                     np.array(ch['xyz']['max']), 24)
        rot[a:b] = decode_quat(arrays['rot'][a:b])
        scale[a:b] = dequantise_linear(arrays['scale'][a:b], np.array(ch['scale']['min']),
                                       np.array(ch['scale']['max']), 16)
        opacity[a:b] = dequantise_linear(arrays['opacity'][a:b], ch['opacity']['min'], ch['opacity']['max'], 16)
        dc[a:b] = dequantise_linear(arrays['dc'][a:b], np.array(ch['dc']['min']), np.array(ch['dc']['max']), 16)
        if k_sh:
            bits = ch['sh']['bits']
            sh[a:b] = dequantise_linear(arrays['sh'][a:b], np.array(ch['sh']['min']),
                                        np.array(ch['sh']['max']), bits)
    return {
        'xyz': xyz.astype(np.float32), 'rot': rot.astype(np.float32), 'log_scale': scale.astype(np.float32),
        'opacity': opacity.astype(np.float32), 'f_dc': dc.astype(np.float32), 'f_rest': sh.astype(np.float32),
    }


def error_report(exact, dec):
    """Errors of a decoded GaussianSet against the exact one (both in source order)."""
    diff_xyz = exact.xyz.astype(np.float64) - dec.xyz.astype(np.float64)
    size = np.exp(exact.log_scale.astype(np.float64).max(axis=1))   # largest axis standard deviation
    pos_err = np.linalg.norm(diff_xyz, axis=1)
    # Angle of the relative rotation. arccos(dot) would be ill-conditioned near dot = 1 (3e-8 of
    # rounding in the dot product becomes 6e-4 rad), so use the vector part of q_exact * conj(q_dec).
    a = exact.rot.astype(np.float64)
    b = dec.rot.astype(np.float64)
    rel_w = a[:, 0] * b[:, 0] + np.sum(a[:, 1:] * b[:, 1:], axis=1)
    rel_v = np.linalg.norm(b[:, :1] * a[:, 1:] - a[:, :1] * b[:, 1:]
                           - np.cross(a[:, 1:], b[:, 1:]), axis=1)
    angle = 2.0 * np.arctan2(rel_v, np.abs(rel_w))
    dscale = np.abs(exact.log_scale.astype(np.float64) - dec.log_scale.astype(np.float64))
    dop = np.abs(exact.opacity.astype(np.float64) - dec.opacity.astype(np.float64))
    op_prob = np.abs(1 / (1 + np.exp(-exact.opacity.astype(np.float64)))
                     - 1 / (1 + np.exp(-dec.opacity.astype(np.float64))))
    ddc = np.abs(exact.f_dc.astype(np.float64) - dec.f_dc.astype(np.float64))
    rep = {
        'pos_abs_max': float(pos_err.max()),
        'pos_rel_splat_p99': float(np.percentile(pos_err / size, 99)),
        'pos_rel_splat_max': float((pos_err / size).max()),
        'rot_angle_max_rad': float(angle.max()),
        'scale_log_abs_max': float(dscale.max()),
        'opacity_logit_abs_max': float(dop.max()),
        'opacity_prob_abs_max': float(op_prob.max()),
        'dc_abs_max': float(ddc.max()),
    }
    if exact.f_rest.shape[1]:
        dsh = np.abs(exact.f_rest.astype(np.float64) - dec.f_rest.astype(np.float64))
        rms_sh = float(np.sqrt(np.mean(exact.f_rest.astype(np.float64) ** 2)))
        rep['sh_abs_max'] = float(dsh.max())
        rep['sh_rms_err_over_rms_signal'] = float(np.sqrt(np.mean(dsh ** 2)) / rms_sh) if rms_sh else 0.0
    return rep
