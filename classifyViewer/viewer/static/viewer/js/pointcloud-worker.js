// =====================================================================
// Worker of the chunked point cloud loader (see pointcloud-loader.js and
// docs/POINTCLOUD_FORMAT.md). Fetches Range slices of geom.bin / col/*.bin and
// decodes them off the main thread. Results are handed back with a transfer
// list, so no copy happens when they cross back to the main thread.
//
// Messages (all answered with { id, ok, result | error }):
//   { type: 'init', cfg }                       cfg: { scale, offset, qMin, bbMin }
//   { type: 'geom', url, start, count, segments, colors }
//        start/count in POINTS (20 B records). segments: [{ key, from, count }] with `from`
//        relative to `start`; optional seg.lod = { model, seqStart } adds lodInfo (Float32 pairs: lodSpacing,
//        fullSpacing of the point, see pointcloud-lod.js). Result: { segments: [{ key, positions, colors?, pointIds, lodInfo? }] }
//   { type: 'column', url, start, count, kind, segments }
//        kind: 'float32' | 'uint8'. Result: { segments: [{ key, values }] } where values is a
//        Float32Array and missing data (NaN / 255) is already mapped to FEATURE_MISSING_SENTINEL.
// =====================================================================

import { fillLodInfo } from './pointcloud-lod.js';

const GEOM_RECORD = 20;
const FEATURE_MISSING_SENTINEL = -1e38;
const UINT8_MISSING = 255;

let cfg = null;

async function fetchRange(url, firstByte, byteLength) {
    const lastByte = firstByte + byteLength - 1;
    const response = await fetch(url, { headers: { 'Range': `bytes=${firstByte}-${lastByte}` } });
    if (response.status === 206) return await response.arrayBuffer();
    if (response.ok) {
        // The server ignored the Range header: slice locally
        const full = await response.arrayBuffer();
        return full.slice(firstByte, firstByte + byteLength);
    }
    throw new Error(`Range request failed (${response.status}) ${url}`);
}

function decodeGeometry(buffer, from, count, withColors) {
    const available = Math.floor(buffer.byteLength / GEOM_RECORD);
    const n = Math.max(0, Math.min(count, available - from));
    const i32 = new Int32Array(buffer, 0, Math.floor(buffer.byteLength / 4));
    const u8 = new Uint8Array(buffer);
    const positions = new Float32Array(n * 3);
    const pointIds = new Int32Array(n);
    const colors = withColors ? new Float32Array(n * 4) : null;

    const [sx, sy, sz] = cfg.scale;
    const [qx, qy, qz] = cfg.qMin;
    // (q + qMin) * scale + offset - bbMin, evaluated in double precision
    const cx = qx * sx + cfg.offset[0] - cfg.bbMin[0];
    const cy = qy * sy + cfg.offset[1] - cfg.bbMin[1];
    const cz = qz * sz + cfg.offset[2] - cfg.bbMin[2];

    for (let j = 0; j < n; j++) {
        const rec = from + j;
        const w = rec * 5;
        positions[3 * j] = i32[w] * sx + cx;
        positions[3 * j + 1] = i32[w + 1] * sy + cy;
        positions[3 * j + 2] = i32[w + 2] * sz + cz;
        pointIds[j] = i32[w + 4];
        if (withColors) {
            const b = rec * GEOM_RECORD + 12;
            colors[4 * j] = u8[b] / 255.0;
            colors[4 * j + 1] = u8[b + 1] / 255.0;
            colors[4 * j + 2] = u8[b + 2] / 255.0;
            colors[4 * j + 3] = 1.0;
        }
    }
    return { positions, colors, pointIds };
}

function decodeColumn(buffer, from, count, kind) {
    const elemSize = kind === 'uint8' ? 1 : 4;
    const available = Math.floor(buffer.byteLength / elemSize);
    const n = Math.max(0, Math.min(count, available - from));
    const values = new Float32Array(n);
    if (kind === 'uint8') {
        const u8 = new Uint8Array(buffer);
        for (let j = 0; j < n; j++) {
            const v = u8[from + j];
            values[j] = v === UINT8_MISSING ? FEATURE_MISSING_SENTINEL : v;
        }
    } else {
        const f32 = new Float32Array(buffer, 0, Math.floor(buffer.byteLength / 4));
        for (let j = 0; j < n; j++) {
            const v = f32[from + j];
            values[j] = Number.isFinite(v) ? v : FEATURE_MISSING_SENTINEL;
        }
    }
    return values;
}

self.onmessage = async (e) => {
    const m = e.data;
    try {
        if (m.type === 'init') {
            cfg = m.cfg;
            self.postMessage({ id: m.id, ok: true, result: null });
            return;
        }

        if (m.type === 'geom') {
            const buffer = await fetchRange(m.url, m.start * GEOM_RECORD, m.count * GEOM_RECORD);
            const segments = [];
            const transfer = [];
            for (const seg of m.segments) {
                const d = decodeGeometry(buffer, seg.from, seg.count, m.colors !== false);
                const out = { key: seg.key, positions: d.positions, pointIds: d.pointIds };
                transfer.push(d.positions.buffer, d.pointIds.buffer);
                if (d.colors) { out.colors = d.colors; transfer.push(d.colors.buffer); }
                if (seg.lod) {
                    const n = d.positions.length / 3;
                    const info = new Float32Array(2 * n);
                    fillLodInfo(seg.lod.model, seg.lod.seqStart, n, info);
                    out.lodInfo = info; transfer.push(info.buffer);
                }
                segments.push(out);
            }
            self.postMessage({ id: m.id, ok: true, result: { segments } }, transfer);
            return;
        }

        if (m.type === 'column') {
            const elemSize = m.kind === 'uint8' ? 1 : 4;
            const buffer = await fetchRange(m.url, m.start * elemSize, m.count * elemSize);
            const segments = [];
            const transfer = [];
            for (const seg of m.segments) {
                const values = decodeColumn(buffer, seg.from, seg.count, m.kind);
                segments.push({ key: seg.key, values });
                transfer.push(values.buffer);
            }
            self.postMessage({ id: m.id, ok: true, result: { segments } }, transfer);
            return;
        }

        throw new Error(`Unknown message type: ${m.type}`);
    } catch (err) {
        self.postMessage({ id: m.id, ok: false, error: String(err && err.message ? err.message : err) });
    }
};
