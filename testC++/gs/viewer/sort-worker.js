// Depth sort for the gs test renderer, off the main thread.
//   init: { type: 'init', n, xyz }   xyz = Float32Array(3n), positions (origin-relative), transferred
//   sort: { type: 'sort', id, view } view = column-major 4x4 view matrix (Float32Array(16))
// Replies: { type: 'ready', order } once, then { type: 'order', id, order, ms } per sort.
// order = splat indices in ascending view-space z (farthest first), a two-pass 16-bit LSD radix sort.

let n = 0;
let xyz = null;
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

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') {
    n = m.n;
    xyz = m.xyz;
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
    const order = sorted.slice();          // work buffers are reused, so send a copy
    const ms = performance.now() - t0;
    self.postMessage({ type: 'order', id: m.id, order, ms }, [order.buffer]);
  }
};
