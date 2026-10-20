// Depth sort and LOD selection for the gs splat layer, off the main thread.
//   init: { type: 'init', n, xyz, starts }   xyz = Float32Array(3n) positions relative to the median (transferred);
//                                            starts = Uint32Array(chunks + 1), first row of every chunk, then n
//   sort: { type: 'sort', id, view, keep }   view = column-major view matrix; keep = Uint32Array(chunks), how many
//                                            rows of each chunk to draw (a prefix: the rows are shuffled per chunk)
// Replies: { type: 'ready', order } once (identity), then { type: 'order', id, order, ms } per sort.
// order = the drawn splat rows in ascending view-space z (farthest first). Radix sort, two 16-bit passes.

let n = 0;
let xyz = null;
let starts = null;
let chunkOf = null;
let depth = null;
let keys = null;
let bufA = null;
let bufB = null;
const count = new Uint32Array(65536);

function sortByView(view) {
    const v2 = view[2], v6 = view[6], v10 = view[10], v14 = view[14];
    for (let i = 0; i < n; i++) {
        const o = 3 * i;
        depth[i] = v2 * xyz[o] + v6 * xyz[o + 1] + v10 * xyz[o + 2] + v14;
    }
    // float32 bits -> unsigned keys that sort like the floats
    const u = new Uint32Array(depth.buffer);
    for (let i = 0; i < n; i++) {
        const b = u[i];
        keys[i] = (b & 0x80000000) ? (~b >>> 0) : ((b | 0x80000000) >>> 0);
    }
    let src = bufA, dst = bufB;
    for (let i = 0; i < n; i++) src[i] = i;
    for (const shift of [0, 16]) {
        count.fill(0);
        for (let i = 0; i < n; i++) count[(keys[i] >>> shift) & 0xffff]++;
        let sum = 0;
        for (let d = 0; d < 65536; d++) { const c = count[d]; count[d] = sum; sum += c; }
        for (let i = 0; i < n; i++) {
            const id = src[i];
            dst[count[(keys[id] >>> shift) & 0xffff]++] = id;
        }
        [src, dst] = [dst, src];
    }
    return src;
}

// Keeps the rows that are in the drawn prefix of their chunk (rank = row - chunk start < keep[chunk]).
function selectRows(sorted, keep) {
    const out = new Uint32Array(n);
    let m = 0;
    for (let k = 0; k < n; k++) {
        const id = sorted[k];
        const c = chunkOf[id];
        if (id - starts[c] < keep[c]) out[m++] = id;
    }
    return out.slice(0, m);
}

self.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'init') {
        n = m.n;
        xyz = m.xyz;
        starts = m.starts;
        const chunks = starts.length - 1;
        chunkOf = new Uint32Array(n);
        for (let c = 0; c < chunks; c++) chunkOf.fill(c, starts[c], starts[c + 1]);
        depth = new Float32Array(n);
        keys = new Uint32Array(n);
        bufA = new Uint32Array(n);
        bufB = new Uint32Array(n);
        const identity = new Uint32Array(n);
        for (let i = 0; i < n; i++) identity[i] = i;
        self.postMessage({ type: 'ready', order: identity }, [identity.buffer]);
    } else if (m.type === 'sort') {
        const t0 = performance.now();
        const sorted = sortByView(m.view);
        const order = selectRows(sorted, m.keep);
        const ms = performance.now() - t0;
        self.postMessage({ type: 'order', id: m.id, order, ms }, [order.buffer]);
    }
};
