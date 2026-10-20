"""Reference renderer for the gs test viewer, in numpy, written from the formulas and not from the shader.

  3DGS: EWA splatting. Covariance in camera space W * Sigma * W^T, projected with the perspective Jacobian
        (clamped to 1.3 of the frustum), 0.3 px low-pass term, footprint = the oriented 3 sigma rectangle.
  2DGS: each splat is a disc. For each pixel centre the ray is intersected with the disc plane, the Gaussian
        weight exp(-0.5 (s^2 + t^2)) is taken in the disc coordinates, cut at 3 sigma.
Both: SH degree 3 colour from the view direction, opacity = sigmoid, back-to-front "over" compositing on the
same background as the browser page.

It reads the same gs v2 folder the browser reads (decoded by gs_format.read_gs, file order), takes the
splats with row % stride == 0 (the same subset as the browser), and uses the same camera as the page:
    eye = median + dist * (cos p sin y, sin p, cos p cos y), looking at the median (origin-relative
    coordinates are relative to the median, as in index.html), up = +y, vertical fov 50 degrees.

Usage:
  python reference_render.py FOLDER OUT.png --w 900 --h 600 --yaw 0.6 --pitch 0.35 --dist 1.2 \
      --median X Y Z --stride 8
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'classifyViewer', 'viewer', 'utils_functions'))

import gs_format  # noqa: E402

BG = np.array([0.09, 0.094, 0.11])
FOV = np.deg2rad(50.0)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_colour(f_dc, f_rest, dirs):
    """Colour per splat from SH degree 3 (INRIA convention), dirs = unit view directions (N, 3)."""
    n = len(f_dc)
    coeffs = np.zeros((n, 3, 16))
    coeffs[:, :, 0] = f_dc
    if f_rest.shape[1]:
        coeffs[:, :, 1:] = f_rest.reshape(n, 3, 15)     # channel-major
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    basis = [
        np.full_like(x, SH_C0),
        -SH_C1 * y, SH_C1 * z, -SH_C1 * x,
        SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * xz, SH_C2[4] * (xx - yy),
        SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z, SH_C3[2] * y * (4 * zz - xx - yy),
        SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy),
        SH_C3[5] * z * (xx - yy), SH_C3[6] * x * (xx - 3 * yy),
    ]
    col = np.zeros((n, 3))
    for k, b in enumerate(basis):
        col += coeffs[:, :, k] * b
    return np.maximum(col + 0.5, 0.0)


def rotation(q):
    """Rotation matrices (N, 3, 3) from unit quaternions (w, x, y, z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def camera(eye, target):
    """Rows x, y, z of the camera basis, z pointing back from the target (as lookAt in index.html)."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def render(gs, w, h, yaw, pitch, dist, median):
    target = np.zeros(3)
    eye = target + dist * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    B = camera(eye, target)
    focal = (h / 2) / np.tan(FOV / 2)
    xyz = gs.xyz.astype(np.float64) - median
    tv = (xyz - eye) @ B.T                       # camera space, z < 0 in front
    colour = sh_colour(gs.f_dc.astype(np.float64), gs.f_rest.astype(np.float64),
                       (xyz - eye) / np.linalg.norm(xyz - eye, axis=1)[:, None])
    alpha = 1.0 / (1.0 + np.exp(-gs.opacity.astype(np.float64)))
    R = rotation(gs.rot.astype(np.float64))
    S = np.exp(gs.log_scale.astype(np.float64))

    keep = tv[:, 2] < -0.05
    order = np.argsort(tv[:, 2])                 # most negative z first = farthest first
    order = order[keep[order]]

    img = np.tile(BG, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    px = xx + 0.5
    py = yy + 0.5

    if gs.kind == '2dgs':
        # disc axes in world space: U = R e1 * s0, V = R e2 * s1, normal n = U x V
        U = R[:, :, 0] * S[:, 0:1]
        V = R[:, :, 1] * S[:, 1:2]
        nrm = np.cross(U, V)
        nrm = nrm / np.linalg.norm(nrm, axis=1)[:, None]
        # pixel rays in world space: camera-space direction (ndc, -1) rotated back with B^T
        dx = (px - w / 2) / focal
        dy = -(py - h / 2) / focal
        dcam = np.stack([dx, dy, -np.ones_like(dx)], -1)          # (h, w, 3)
        dworld = dcam @ B                                          # rows of B^T applied: B^T v
        dworld = dworld / np.linalg.norm(dworld, axis=-1, keepdims=True)
        for i in order:
            c = xyz[i]
            # screen bounding box of the 3 sigma square of the disc
            corners = np.array([c + a * 3 * U[i] + b * 3 * V[i] for a in (-1, 1) for b in (-1, 1)])
            cc = (corners - eye) @ B.T
            if np.any(cc[:, 2] >= -0.05):
                continue
            sx = w / 2 + focal * cc[:, 0] / -cc[:, 2]
            sy = h / 2 - focal * cc[:, 1] / -cc[:, 2]
            x0, x1 = max(0, int(np.floor(sx.min()))), min(w, int(np.ceil(sx.max())) + 1)
            y0, y1 = max(0, int(np.floor(sy.min()))), min(h, int(np.ceil(sy.max())) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            d = dworld[y0:y1, x0:x1]
            den = d @ nrm[i]
            ok = np.abs(den) > 1e-12
            t = np.where(ok, ((c - eye) @ nrm[i]) / np.where(ok, den, 1.0), 0.0)
            p = eye + t[..., None] * d - c
            su = (p @ U[i]) / (U[i] @ U[i])
            sv = (p @ V[i]) / (V[i] @ V[i])
            g = -0.5 * (su * su + sv * sv)
            a = np.minimum(0.99, alpha[i] * np.exp(g)) * (ok & (g >= -4.5) & (t > 0))
            a = np.where(a < 1.0 / 255.0, 0.0, a)[..., None]
            img[y0:y1, x0:x1] = colour[i] * a + img[y0:y1, x0:x1] * (1 - a)
    else:
        for i in order:
            t = tv[i]
            tz = t[2]
            limx = 1.3 * (w / 2) / focal
            limy = 1.3 * (h / 2) / focal
            tx = np.clip(t[0] / tz, -limx, limx) * tz
            ty = np.clip(t[1] / tz, -limy, limy) * tz
            J = np.array([[focal / tz, 0.0, -focal * tx / (tz * tz)],
                          [0.0, focal / tz, -focal * ty / (tz * tz)]])
            Sig = (R[i] * S[i][None, :] ** 2) @ R[i].T           # world covariance
            cov3 = B @ Sig @ B.T                                  # camera space
            cov2 = J @ cov3 @ J.T
            a_, b_, c_ = cov2[0, 0] + 0.3, cov2[0, 1], cov2[1, 1] + 0.3
            det = a_ * c_ - b_ * b_
            if det <= 0:
                continue
            mid = 0.5 * (a_ + c_)
            disc = np.sqrt(max(0.1, mid * mid - det))
            l1, l2 = mid + disc, mid - disc
            ev, evec = np.linalg.eigh(np.array([[a_, b_], [b_, c_]]))
            major, minor = evec[:, 1], evec[:, 0]                 # eigenvalues ascending
            r1, r2 = 3 * np.sqrt(l1), 3 * np.sqrt(max(l2, 0.0))
            # the centre is projected unclamped, as the vertex shader does
            sx = w / 2 + focal * t[0] / -tz
            sy = h / 2 - focal * t[1] / -tz
            rad = np.hypot(r1, r2)
            x0, x1 = max(0, int(np.floor(sx - rad))), min(w, int(np.ceil(sx + rad)) + 1)
            y0, y1 = max(0, int(np.floor(sy - rad))), min(h, int(np.ceil(sy + rad)) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            # pixel offsets in the screen frame (y up in the covariance, y down in pixels)
            dx = px[y0:y1, x0:x1] - sx
            dy = -(py[y0:y1, x0:x1] - sy)
            u = dx * major[0] + dy * major[1]
            v = dx * minor[0] + dy * minor[1]
            inside = (np.abs(u) <= r1) & (np.abs(v) <= r2)
            cov_inv = np.array([[c_, -b_], [-b_, a_]]) / det
            power = -0.5 * (cov_inv[0, 0] * dx * dx + 2 * cov_inv[0, 1] * dx * dy + cov_inv[1, 1] * dy * dy)
            a = np.minimum(0.99, alpha[i] * np.exp(power)) * (inside & (power <= 0))
            a = np.where(a < 1.0 / 255.0, 0.0, a)[..., None]
            img[y0:y1, x0:x1] = colour[i] * a + img[y0:y1, x0:x1] * (1 - a)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('out')
    ap.add_argument('--w', type=int, default=900)
    ap.add_argument('--h', type=int, default=600)
    ap.add_argument('--yaw', type=float, default=0.6)
    ap.add_argument('--pitch', type=float, default=0.35)
    ap.add_argument('--dist', type=float, required=True)
    ap.add_argument('--median', type=float, nargs=3, required=True)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    full = gs_format.read_gs(args.folder, source_order=False)   # file order, as the browser reads it
    idx = np.arange(0, len(full), args.stride)
    sub = gs_format.GaussianSet(
        kind=full.kind, source=full.source, xyz=full.xyz[idx], rot=full.rot[idx],
        log_scale=full.log_scale[idx], opacity=full.opacity[idx], f_dc=full.f_dc[idx],
        f_rest=full.f_rest[idx] if full.f_rest.shape[1] else full.f_rest, sh_degree=full.sh_degree)
    img = render(sub, args.w, args.h, args.yaw, args.pitch, args.dist, np.array(args.median))
    Image.fromarray(img).save(args.out)
    print(f'reference {args.out}: {len(idx)} splats ({sub.kind})')


if __name__ == '__main__':
    main()
