// Gaussian splat layer for the main viewer (Babylon.js).
//
// Loads a `gs` v2 folder (see docs/GAUSSIAN_SPLAT_FORMAT.md, produced by viewer/gs_import.py), decodes it, sorts the
// drawn splats by view depth in gs-sort-worker.js and draws them as instanced quads in one ShaderMaterial:
//   3DGS: EWA projection of the 3D covariance, footprint = 3 sigma;  2DGS: disc, ray / plane intersection.
// The colour is SH degree 3, the blending is premultiplied, back to front.
//
// LOD (per chunk, see chooseKeep): a chunk far away, or small on screen, draws a prefix of its rows. The rows of a
// chunk are shuffled at write time, so a prefix is a uniform sample. The kept splats are scaled by f^-1/2 (screen
// coverage preserved), which the shader applies from a per-chunk table.
//
// Limits: perspective camera only (orthographic hides the layer), no editing / classes / export, the splats are not
// moved by the point cloud root transform.

const TEXELS = 15;      // texels per splat: 0 centre+opacity, 1-2 covariance or disc axes (+ chunk index), 3-14 SH
const TEX_W = 4096;     // one splat per column; its 15 texels are the rows k + 15 * block
const SMALL = 1 / Math.SQRT2;
const LOD_TARGET = 1.5; // width in pixels of the drawn splats (std dev): splats thinner than this are thinned
const LOD_MIN = 0.1;    // smallest fraction of a chunk that is drawn (scale compensation at most ~3.2x)
const NEAR = 0.05;      // splats whose centre is nearer than this to the camera plane are dropped by the vertex shader
const CULL_GUARD = 1.2; // chunks are culled against a view cone this much wider: the drawn set is only refreshed when a sort
                        // arrives (tens of ms), so a chunk that enters the view during a fast rotation must already be in it

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Column-major perspective matrix (GL convention).
function perspectiveArray(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1,
        0, 0, 2 * far * near / (near - far), 0]);
}

// Decodes the gs v2 columns into the GPU layout. Positions are relative to the median (float32 precision).
async function decodeFolder(base) {
    const meta = await (await fetch(`${base}/meta.json`, { cache: 'no-store' })).json();
    if (meta.format !== 'gs' || meta.formatVersion !== 2) throw new Error('splat folder needs gs formatVersion 2');
    const n = meta.count;
    const col = async (key) => new Uint8Array(await (await fetch(`${base}/${meta.columns[key].file}`, { cache: 'no-store' })).arrayBuffer());
    const xyzB = await col('xyz');
    const rotB = await col('rot');
    const scaleB = await col('scale');
    const opB = await col('opacity');
    const dcB = await col('dc');
    const shB = meta.columns.sh ? await col('sh') : null;
    const rot16 = new Uint16Array(rotB.buffer, rotB.byteOffset, n * 4);
    const K = meta.columns.scale.components;
    const sc16 = new Uint16Array(scaleB.buffer, scaleB.byteOffset, n * K);
    const op16 = new Uint16Array(opB.buffer, opB.byteOffset, n);
    const dc16 = new Uint16Array(dcB.buffer, dcB.byteOffset, n * 3);
    const is2d = meta.kind === '2dgs';
    const nChunks = meta.chunks.length;
    const chunkLogSum = new Float64Array(nChunks);

    const blocks = Math.ceil(n / TEX_W);
    const texH = TEXELS * blocks;
    const tex = new Float32Array(TEX_W * texH * 4);
    const q4 = new Float64Array(4);
    const xyzRaw = new Float64Array(3 * n);
    meta.chunks.forEach((ch, ci) => {
        const lo = ch.xyz.min, hi = ch.xyz.max;
        const sMin = ch.scale.min, sMax = ch.scale.max;
        const oMin = ch.opacity.min, oMax = ch.opacity.max;
        const dMin = ch.dc.min, dMax = ch.dc.max;
        const shMin = ch.sh ? ch.sh.min : null, shMax = ch.sh ? ch.sh.max : null;
        const shQ = ch.sh ? (1 << ch.sh.bits) - 1 : 255;
        const sp = [0, 1, 2].map((j) => (hi[j] - lo[j]) || 1);
        const sSpan = [0, 1, 2].map((j) => (sMax[j] - sMin[j]) || 1);
        const dSpan = [0, 1, 2].map((j) => (dMax[j] - dMin[j]) || 1);
        const shSpan = shMin ? shMin.map((m, k) => (shMax[k] - m) || 1) : null;
        const oSpan = (oMax - oMin) || 1;
        for (let r = 0; r < ch.count; r++) {
            const i = ch.start + r;
            const colX = i % TEX_W, block = Math.floor(i / TEX_W);
            const at = (k) => ((TEXELS * block + k) * TEX_W + colX) * 4;   // float offset of texel k of row i

            const px = [0, 0, 0];
            for (let j = 0; j < 3; j++) {
                const o = i * 9 + j * 3;
                const q = xyzB[o] | (xyzB[o + 1] << 8) | (xyzB[o + 2] << 16);
                px[j] = lo[j] + sp[j] * (q / 16777215);
                xyzRaw[3 * i + j] = px[j];
            }
            decodeQuat(rot16[i * 4], rot16[i * 4 + 1], rot16[i * 4 + 2], rot16[i * 4 + 3], q4);
            const s = [0, 0, 0];
            for (let j = 0; j < K; j++) s[j] = sMin[j] + sSpan[j] * (sc16[i * K + j] / 65535);
            chunkLogSum[ci] += Math.max(...s.slice(0, K));
            const opacity = oMin + oSpan * (op16[i] / 65535);
            const dc = [0, 1, 2].map((j) => dMin[j] + dSpan[j] * (dc16[i * 3 + j] / 65535));

            const o0 = at(0);
            tex[o0] = px[0]; tex[o0 + 1] = px[1]; tex[o0 + 2] = px[2]; tex[o0 + 3] = sigmoid(opacity);

            const [qw, qx, qy, qz] = q4;
            const c0 = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)];
            const c1 = [2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)];
            const c2 = [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)];
            const a1 = at(1), a2 = at(2);
            if (is2d) {
                const U = c0.map((v) => v * Math.exp(s[0]));
                const V = c1.map((v) => v * Math.exp(s[1]));
                tex[a1] = U[0]; tex[a1 + 1] = U[1]; tex[a1 + 2] = U[2];
                tex[a2] = V[0]; tex[a2 + 1] = V[1]; tex[a2 + 2] = V[2];
            } else {
                // Sigma = sum_j (R_j s_j)(R_j s_j)^T, upper triangle
                let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
                for (const [j, c] of [c0, c1, c2].entries()) {
                    const m0 = c[0] * Math.exp(s[j]), m1 = c[1] * Math.exp(s[j]), m2 = c[2] * Math.exp(s[j]);
                    xx += m0 * m0; xy += m0 * m1; xz += m0 * m2; yy += m1 * m1; yz += m1 * m2; zz += m2 * m2;
                }
                tex[a1] = xx; tex[a1 + 1] = xy; tex[a1 + 2] = xz; tex[a1 + 3] = yy;
                tex[a2] = yz; tex[a2 + 1] = zz;
            }
            tex[a2 + 3] = ci;                       // chunk index, used by the shader for the LOD compensation

            for (let c = 0; c < 3; c++) {
                // channel c: 4 texels in consecutive rows (4 floats each) = 16 values, DC first
                const base = (TEXELS * block + 3 + c * 4) * TEX_W + colX;
                const rowFloat = (kt) => (base + kt * TEX_W) * 4;
                for (let k = 0; k < 16; k++) {
                    let v;
                    if (k === 0) v = dc[c];
                    else if (shB) {
                        const kk = c * 15 + (k - 1);
                        v = shMin[kk] + shSpan[kk] * (shB[i * 45 + kk] / shQ);
                    } else v = 0;
                    tex[rowFloat(k >> 2) + (k & 3)] = v;
                }
            }
        }
        });
    // centre on the median position: float32 keeps its precision for far floaters
    const sampleStep = Math.max(1, Math.floor(n / 20000));
    const xs = [];
    for (let i = 0; i < n; i += sampleStep) xs.push([xyzRaw[3 * i], xyzRaw[3 * i + 1], xyzRaw[3 * i + 2]]);
    const median = [0, 1, 2].map((j) => { const v = xs.map((p) => p[j]).sort((a, b) => a - b); return v[v.length >> 1]; });
    for (let i = 0; i < n; i++) {
        const colX = i % TEX_W, block = Math.floor(i / TEX_W);
        const o = ((TEXELS * block) * TEX_W + colX) * 4;
        tex[o] -= median[0]; tex[o + 1] -= median[1]; tex[o + 2] -= median[2];
    }
    // distances for the initial camera, and per-chunk LOD data (centre and radius relative to the median)
    const dists = [];
    for (let i = 0; i < n; i += sampleStep) {
        dists.push(Math.hypot(xyzRaw[3 * i] - median[0], xyzRaw[3 * i + 1] - median[1], xyzRaw[3 * i + 2] - median[2]));
    }
    dists.sort((a, b) => a - b);
    const radius = dists[Math.floor(dists.length * 0.9)] || 1;
    const chunks = meta.chunks.map((ch, ci) => {
        const mid = [0, 1, 2].map((j) => (ch.min[j] + ch.max[j]) / 2 - median[j]);
        const half = Math.hypot(...[0, 1, 2].map((j) => (ch.max[j] - ch.min[j]) / 2));
        return {
            start: ch.start, count: ch.count, centre: mid, radius: half,
            size: Math.exp(chunkLogSum[ci] / ch.count),     // geometric mean of the largest axis of the chunk
        };
    });
    const xyz = xyzCopyOf(tex, n);
    return { meta, n, tex, texH, median, radius, is2d, xyz, chunks, lodAllowed: meta.order === 'morton-chunk-random' };
}

// Positions relative to the median, for the sort worker (3 floats per splat).
function xyzCopyOf(tex, n) {
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        const colX = i % TEX_W, block = Math.floor(i / TEX_W);
        const o = ((TEXELS * block) * TEX_W + colX) * 4;
        out[3 * i] = tex[o]; out[3 * i + 1] = tex[o + 1]; out[3 * i + 2] = tex[o + 2];
    }
    return out;
}

// Smallest-three quaternion (same as gs_quant.decode_quat). Output rot = w, x, y, z normalised.
function decodeQuat(p0, p1, p2, idx, out) {
    const a = (p0 / 65535 * 2 - 1) * SMALL;
    const b = (p1 / 65535 * 2 - 1) * SMALL;
    const c = (p2 / 65535 * 2 - 1) * SMALL;
    const w = Math.sqrt(Math.max(0, 1 - (a * a + b * b + c * c)));
    const q = [0, 0, 0, 0];
    if (idx === 0) { q[0] = w; q[1] = a; q[2] = b; q[3] = c; }
    else if (idx === 1) { q[0] = a; q[1] = w; q[2] = b; q[3] = c; }
    else if (idx === 2) { q[0] = a; q[1] = b; q[2] = w; q[3] = c; }
    else { q[0] = a; q[1] = b; q[2] = c; q[3] = w; }
    const nn = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    out[0] = q[0] / nn; out[1] = q[1] / nn; out[2] = q[2] / nn; out[3] = q[3] / nn;
}

// LOD policy. For every chunk: cull it when it is outside the view, otherwise keep a fraction f of its rows where
// f = LOD_TARGET / (splats per pixel the whole chunk would put on the screen), clamped to [LOD_MIN, 1].
// V is the column-major view matrix of the splats (origin-relative, z flipped: the camera looks down -z).
function chooseKeep(V, chunks, focal, fovy, aspect, lodOn) {
    const nC = chunks.length;
    const keep = new Uint32Array(nC);
    const frac = new Float32Array(nC);
    const tanY = Math.tan(fovy / 2), tanX = tanY * aspect;
    const gx = tanX * CULL_GUARD, gy = tanY * CULL_GUARD;                // slopes of the culling cone
    const sx = Math.sqrt(1 + gx * gx), sy = Math.sqrt(1 + gy * gy);
    for (let c = 0; c < nC; c++) {
        const ch = chunks[c];
        const [x, y, z] = ch.centre;
        const tx = V[0] * x + V[4] * y + V[8] * z + V[12];
        const ty = V[1] * x + V[5] * y + V[9] * z + V[13];
        const tz = V[2] * x + V[6] * y + V[10] * z + V[14];
        const R = ch.radius;
        const dist = -tz;
        // A chunk is culled only when the camera is outside its bounding sphere and the sphere is entirely outside one
        // plane of the view (a sphere that contains the camera always has splats in view). Camera space looks down -z:
        //   near : the FRONT of the sphere (tz - R) is still behind the near plane. The test used to be tz + R, the back of
        //          the sphere: it culled every chunk whose sphere reaches behind the camera although its centre is in front.
        //   sides: signed distance of the centre to the plane, compared with -R. A plane through the camera with slope
        //          tan has the unit normal (+-1, 0, -tan) / sqrt(1 + tan^2): the sphere reaches R * sqrt(1 + tan^2) beyond
        //          the line x = dist * tan, not R.
        const outside = Math.hypot(tx, ty, tz) > R;
        if (outside && (tz - R > -NEAR
            || (-tx - tz * gx) < -R * sx || (tx - tz * gx) < -R * sx
            || (-ty - tz * gy) < -R * sy || (ty - tz * gy) < -R * sy)) {
            keep[c] = 0; frac[c] = LOD_MIN;
            continue;
        }
        let f = 1;
        if (lodOn) {
            // size rule: a typical splat of the chunk is p pixels wide. Keeping a fraction f and scaling the kept splats
            // by f^-1/2 gives them a width of p / sqrt(f), so f = (p / T)^2 makes them T pixels wide (T = LOD_TARGET).
            const p = focal * ch.size / Math.max(dist - R, 1e-3);          // nearest part of the chunk, px (std dev)
            f = Math.min(1, Math.max(LOD_MIN, (p / LOD_TARGET) ** 2));
            // quarter-octave steps: small camera moves do not change the drawn set, so no new sort is needed
            f = Math.min(1, Math.max(LOD_MIN, 2 ** (Math.round(4 * Math.log2(f)) / 4)));
        }
        frac[c] = f;
        keep[c] = f >= 1 ? ch.count : Math.max(1, Math.ceil(ch.count * f));
    }
    return { keep, frac };
}

// Shaders in GLSL ES 3.00 (Babylon adds the version line and the GL defines).
const VERTEX = `
precision highp float;
precision highp int;
attribute vec3 position;
attribute float aIndex;
uniform sampler2D uData;
uniform sampler2D uChunks;
uniform float uNChunks;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform float uFocal;
uniform vec3 uCam;
uniform float uKind;
uniform float uTexW;
uniform float uTexH;
out vec4 vColor;
out vec2 vOff;
out vec3 vConic;
out vec3 vWorld;
out vec3 vC;
out vec3 vU;
out vec3 vV;

vec4 tex(float idx, float k) {
    float col = mod(idx, uTexW);
    float block = floor(idx / uTexW);
    float y = k + 15.0 * block;
    return texture(uData, (vec2(col, y) + 0.5) / vec2(uTexW, uTexH));
}
void loadCh(float idx, float base, out float s[16]) {
    vec4 a = tex(idx, base);
    vec4 b = tex(idx, base + 1.0);
    vec4 c = tex(idx, base + 2.0);
    vec4 d = tex(idx, base + 3.0);
    s[0] = a.x; s[1] = a.y; s[2] = a.z; s[3] = a.w;
    s[4] = b.x; s[5] = b.y; s[6] = b.z; s[7] = b.w;
    s[8] = c.x; s[9] = c.y; s[10] = c.z; s[11] = c.w;
    s[12] = d.x; s[13] = d.y; s[14] = d.z; s[15] = d.w;
}
float evalChannel(float s[16], vec3 d) {
    float x = d.x, y = d.y, z = d.z;
    float r = 0.28209479177387814 * s[0];
    r += -0.4886025119029199 * y * s[1] + 0.4886025119029199 * z * s[2] - 0.4886025119029199 * x * s[3];
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    r += 1.0925484305920792 * xy * s[4] - 1.0925484305920792 * yz * s[5]
       + 0.31539156525252005 * (2.0 * zz - xx - yy) * s[6]
       - 1.0925484305920792 * xz * s[7] + 0.5462742152960396 * (xx - yy) * s[8];
    r += -0.5900435899266435 * y * (3.0 * xx - yy) * s[9] + 2.890611442640554 * xy * z * s[10]
       - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * s[11]
       + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * s[12]
       - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * s[13]
       + 1.445305721320277 * z * (xx - yy) * s[14] - 0.5900435899266435 * x * (xx - 3.0 * yy) * s[15];
    return r;
}
void main() {
    vec4 p0 = tex(aIndex, 0.0);
    vec3 c = p0.xyz;
    vec3 tv = (uView * vec4(c, 1.0)).xyz;
    if (tv.z > -${NEAR.toFixed(2)}) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); return; }

    vec3 dir = normalize(c - uCam);
    float s[16];
    loadCh(aIndex, 3.0, s);  float cr = evalChannel(s, dir);
    loadCh(aIndex, 7.0, s);  float cg = evalChannel(s, dir);
    loadCh(aIndex, 11.0, s); float cb = evalChannel(s, dir);
    vColor = vec4(max(vec3(cr, cg, cb) + 0.5, 0.0), p0.w);
    vC = c;
    vec2 corner = position.xy;

    // LOD compensation: the chunk's kept fraction f scales the splat by f^-1/2 (screen coverage preserved)
    vec4 c1 = tex(aIndex, 1.0);
    vec4 c2 = tex(aIndex, 2.0);
    float fc = texture(uChunks, vec2((c2.w + 0.5) / uNChunks, 0.5)).r;
    float sc = pow(fc, -0.5);

    if (uKind < 0.5) {
        mat3 Vrk = mat3(c1.x, c1.y, c1.z, c1.y, c1.w, c2.x, c1.z, c2.x, c2.y) * (sc * sc);
        float tz = tv.z;
        float limx = 1.3 * 0.5 * uViewport.x / uFocal;
        float limy = 1.3 * 0.5 * uViewport.y / uFocal;
        float tx = clamp(tv.x / tz, -limx, limx) * tz;
        float ty = clamp(tv.y / tz, -limy, limy) * tz;
        // J is stored transposed: J_std = transpose(J)
        mat3 J = mat3(uFocal / tz, 0.0, -uFocal * tx / (tz * tz),
                      0.0, uFocal / tz, -uFocal * ty / (tz * tz),
                      0.0, 0.0, 0.0);
        mat3 W = mat3(uView);
        mat3 Sc = W * Vrk * transpose(W);
        mat3 cov = transpose(J) * Sc * J;
        float a = cov[0][0] + 0.3;
        float b = cov[0][1];
        float cc = cov[1][1] + 0.3;
        float det = a * cc - b * b;
        if (det <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); return; }
        float mid = 0.5 * (a + cc);
        float disc = sqrt(max(0.1, mid * mid - det));
        float l1 = mid + disc;
        float l2 = mid - disc;
        vec2 v1 = (abs(b) < 1e-6) ? ((a >= cc) ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
                                  : normalize(vec2(b, l1 - a));
        vec2 v2 = vec2(-v1.y, v1.x);
        float r1 = 3.0 * sqrt(l1);
        float r2 = 3.0 * sqrt(max(l2, 0.0));
        vec2 off = corner.x * v1 * r1 + corner.y * v2 * r2;
        vOff = off;
        vConic = vec3(cc, -b, a) / det;
        vec4 clip = uProj * vec4(tv, 1.0);
        gl_Position = vec4(clip.xy + off * 2.0 / uViewport * clip.w, clip.z, clip.w);
    } else {
        vec3 U = c1.xyz * sc;
        vec3 V = c2.xyz * sc;
        vec3 wpos = c + corner.x * 3.0 * U + corner.y * 3.0 * V;
        vWorld = wpos;
        vU = U;
        vV = V;
        gl_Position = uProj * uView * vec4(wpos, 1.0);
    }
}`;

const FRAGMENT = `
precision highp float;
precision highp int;
uniform float uKind;
in vec4 vColor;
in vec2 vOff;
in vec3 vConic;
in vec3 vWorld;
in vec3 vC;
in vec3 vU;
in vec3 vV;
void main() {
    float alpha;
    if (uKind < 0.5) {
        vec2 d = vOff;
        float power = -0.5 * (vConic.x * d.x * d.x + vConic.z * d.y * d.y) - vConic.y * d.x * d.y;
        if (power > 0.0) discard;
        alpha = min(0.99, vColor.a * exp(power));
    } else {
        vec3 p = vWorld - vC;
        float su = dot(p, vU) / dot(vU, vU);
        float sv = dot(p, vV) / dot(vV, vV);
        float g = -0.5 * (su * su + sv * sv);
        if (g < -4.5) discard;
        alpha = min(0.99, vColor.a * exp(g));
    }
    if (alpha < 1.0 / 255.0) discard;
    gl_FragColor = vec4(vColor.rgb * alpha, alpha);
}`;

// Loads a gs v2 folder into the scene. Returns a handle with dispose(). The handle is removed on 'scene-reset'.
export async function loadGaussianSplats(url, scene) {
    const t0 = performance.now();
    const d = await decodeFolder(url);
    const loadMs = performance.now() - t0;
    const BABYLON = window.BABYLON;
    const engine = scene.getEngine();
    const n = d.n;
    const nC = d.chunks.length;
    let lodOn = d.lodAllowed && !new URLSearchParams(location.search).has('nolod');

    const texture = new BABYLON.RawTexture(d.tex, TEX_W, d.texH, BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
        false, false, BABYLON.Texture.NEAREST_SAMPLINGMODE, BABYLON.Engine.TEXTURETYPE_FLOAT);
    const chunkData = new Float32Array(nC * 4).fill(1);
    const chunkTex = new BABYLON.RawTexture(chunkData, nC, 1, BABYLON.Engine.TEXTUREFORMAT_RGBA, scene,
        false, false, BABYLON.Texture.NEAREST_SAMPLINGMODE, BABYLON.Engine.TEXTURETYPE_FLOAT);

    BABYLON.Effect.ShadersStore.gsSplatVertexShader = VERTEX;
    BABYLON.Effect.ShadersStore.gsSplatFragmentShader = FRAGMENT;
    const material = new BABYLON.ShaderMaterial('gsSplatMat', scene, 'gsSplat', {
        attributes: ['position', 'aIndex'],
        uniforms: ['uView', 'uProj', 'uViewport', 'uFocal', 'uCam', 'uKind', 'uTexW', 'uTexH', 'uNChunks'],
        samplers: ['uData', 'uChunks'],
        needAlphaBlending: true,
    });
    material.setTexture('uData', texture);
    material.setTexture('uChunks', chunkTex);
    material.alphaMode = BABYLON.Engine.ALPHA_PREMULTIPLIED;
    material.disableDepthWrite = true;
    material.backFaceCulling = false;

    const quad = new BABYLON.Mesh('gsSplats', scene);
    quad.setVerticesData(BABYLON.VertexBuffer.PositionKind, [-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0], false, 3);
    quad.setIndices([0, 1, 2, 2, 1, 3]);
    quad.material = material;
    quad.isPickable = false;
    quad.alwaysSelectAsActiveMesh = true;

    // instance buffer: the drawn rows; the draw count is forced (Babylon counts only the matrix buffer)
    const order = new Float32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    quad.thinInstanceSetBuffer('aIndex', order, 1, false);
    quad.forcedInstanceCount = n;
    let drawCount = n;

    // sort worker: positions and chunk starts once, then one request per change of view or LOD selection
    const worker = new Worker(new URL('gs-sort-worker.js', import.meta.url));
    const starts = new Uint32Array(nC + 1);
    d.chunks.forEach((ch, c) => { starts[c] = ch.start; });
    starts[nC] = n;
    let sortPending = false, sortId = 0, sortedView = null, sortedFrac = null, requestedView = null;
    let requestedFrac = null, requestedKeep = null;
    let sortMs = 0, sortCount = 0, ready = false;
    let lastState = null;
    worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { ready = true; return; }
        if (m.type !== 'order') return;
        sortPending = false;
        sortMs = m.ms;
        sortCount++;
        sortedView = requestedView;
        sortedFrac = requestedFrac;
        // RGBA texture, one texel per chunk, the shader reads .r: frac[c] goes to float 4c (chunkData.set(requestedFrac) wrote it
        // to float c, so only chunk 0 was ever compensated and every other thinned chunk was drawn with unscaled splats)
        for (let c = 0; c < nC; c++) chunkData[4 * c] = requestedFrac[c];
        chunkTex.update(chunkData);
        const o = m.order;
        drawCount = o.length;
        for (let k = 0; k < drawCount; k++) order[k] = o[k];
        quad.forcedInstanceCount = drawCount;
        quad.thinInstanceBufferUpdated('aIndex');
    };
    worker.postMessage({ type: 'init', n, xyz: d.xyz, starts }, [d.xyz.buffer]);

    // camera: frame the splats on their median
    const camera = scene.activeCamera;
    const median = d.median;
    const medianV = new BABYLON.Vector3(median[0], median[1], median[2]);
    if (camera && camera.setTarget) {
        camera.setTarget(medianV);
        camera.radius = Math.max(0.05, d.radius * 2.6);
    }

    const sameArr = (a, b) => {
        if (!a || !b || a.length !== b.length) return false;
        for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
        return true;
    };
    const F = BABYLON.Matrix.Scaling(1, 1, -1);
    const T = BABYLON.Matrix.Translation(median[0], median[1], median[2]);
    const update = () => {
        const cam = scene.activeCamera;
        if (!cam || cam.mode === BABYLON.Camera.ORTHOGRAPHIC_CAMERA) {
            quad.isVisible = false;
            return;
        }
        quad.isVisible = true;
        // Babylon view (left handed, forward +z) -> our convention (camera looks down -z)
        const V = cam.getViewMatrix(true);
        const Vu = T.multiply(V).multiply(F);
        const fovy = cam.fov;
        const aspect = engine.getAspectRatio(cam);
        const W = engine.getRenderWidth(), H = engine.getRenderHeight();
        const focal = (H / 2) / Math.tan(fovy / 2);
        material.setMatrix('uView', Vu);
        material.setMatrix('uProj', BABYLON.Matrix.FromArray(perspectiveArray(fovy, aspect, 0.01, 1e5)));
        material.setVector2('uViewport', new BABYLON.Vector2(W, H));
        material.setFloat('uFocal', focal);
        const cp = cam.position;
        material.setVector3('uCam', new BABYLON.Vector3(cp.x - median[0], cp.y - median[1], cp.z - median[2]));
        material.setFloat('uKind', d.is2d ? 1 : 0);
        material.setFloat('uTexW', TEX_W);
        material.setFloat('uTexH', d.texH);
        material.setFloat('uNChunks', nC);

        const { keep, frac } = chooseKeep(Vu.m, d.chunks, focal, fovy, aspect, lodOn);
        lastState = { view: Vu.m, focal, fovy, aspect, keep, frac };
        // The compensation table (frac) and the drawn rows (keep) change together, when the sort that uses them
        // arrives: applying the new scale to the old rows (or the reverse) is what leaves holes while LOD changes.
        if (ready && !sortPending && (!sameView(Vu.m, sortedView) || !sameArr(frac, sortedFrac))) {
            sortPending = true;
            sortId++;
            requestedView = Vu.m.slice();
            requestedFrac = frac.slice();
            requestedKeep = keep.slice();
            worker.postMessage({ type: 'sort', id: sortId, view: requestedView, keep: requestedKeep });
        }
    };
    const sameView = (a, b) => {
        if (!a || !b) return false;
        for (let k = 0; k < 16; k++) if (a[k] !== b[k]) return false;
        return true;
    };
    const observer = scene.onBeforeRenderObservable.add(update);

    let disposed = false;
    const dispose = () => {
        if (disposed) return;
        disposed = true;
        scene.onBeforeRenderObservable.remove(observer);
        window.removeEventListener('scene-reset', dispose);
        worker.terminate();
        quad.dispose();
        material.dispose();
        texture.dispose();
        chunkTex.dispose();
        if (scene.gaussianSplats === handle) scene.gaussianSplats = null;
        const countEl = document.getElementById('point-count');
        if (countEl && countEl.textContent === n.toLocaleString()) countEl.textContent = '0';
    };
    window.addEventListener('scene-reset', dispose);

    const handle = {
        n, kind: d.meta.kind, shDegree: d.meta.shDegree, loadMs,
        stats: () => ({ sortMs, sortCount, drawn: drawCount, total: n, lod: lodOn, chunks: nC }),
        setLod: (on) => { lodOn = !!on && d.lodAllowed; },
        debug: () => ({ chunks: d.chunks, state: lastState }),
        dispose,
    };
    scene.gaussianSplats = handle;
    // the info bar of the viewer (POINTS / FPS / VERSION) shows the number of splats, as it shows the points of a point cloud
    const countEl = document.getElementById('point-count');
    if (countEl) countEl.textContent = n.toLocaleString();
    console.log(`gs splats: ${n} (${d.meta.kind}, SH ${d.meta.shDegree}), ${nC} chunks, LOD ${lodOn ? 'on' : 'off'}, decoded in ${loadMs.toFixed(0)} ms`);
    return handle;
}
